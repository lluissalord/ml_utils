import sys
import os
import shutil
import pandas as pd
import numpy as np

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

def extract_zips(zip_path, extract_dir):
    """ Extract zip file """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    shutil.unpack_archive(zip_path, extract_dir=extract_dir)
    print(f"File in {zip_path} extracted in directory {extract_dir}")

def sizeof_fmt(num, suffix='B'):
    """ Transform number to readable unit """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def get_var_list():
    """ Print current variable size in memory """
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

def rebalance_data(df, n_ensemble, target, negative_downsampling=True, use_partial_data=False, ratio_partial_data=3):
    """ Downsampling (negative or positive) data according to ratio and distribute it across number of ensemble """
    pos_rows = df[df[target] == 1].shape[0]
    neg_rows = df[df[target] == 0].shape[0]

    # max_ratio = 2
    # max_n_models = 10
    # min_n_model = (total_rows / ((1 + max_ratio) * pos_rows))

    if negative_downsampling:
        ref_rows = pos_rows
        ref_df = df[df[target] == 1]
        downsampling_rows = neg_rows
        downsampling_df = df[df[target] == 0]
    else:
        ref_rows = neg_rows
        ref_df = df[df[target] == 0]
        downsampling_rows = pos_rows
        downsampling_df = df[df[target] == 1]

    if use_partial_data:
        ratio = ratio_partial_data
        length_downsampling = int(ratio * ref_rows)
    else:
        ratio = (downsampling_rows / (n_ensemble * ref_rows))
        length_downsampling = int(downsampling_rows / n_ensemble)
    print(f'Using {n_ensemble} ensembles, you are setting a ratio of {ratio}.')
    print(f'Each model will be train with {int((1 + ratio) * ref_rows)} samples.')

    all_batch_indexes = pd.DataFrame(columns=range(n_ensemble))
    used_index = np.array([])

    for i in tqdm_notebook(range(n_ensemble)):
        # For downsampling we use random sample of lenght 'length_downsampling' which have not been previously used
        all_batch_indexes[i] = pd.concat([
            ref_df,
            downsampling_df[~downsampling_df.index.isin(used_index)].sample(length_downsampling, random_state=42)
        ]).sort_index().index

        used_index = np.append(used_index, all_batch_indexes[i].values)

    return all_batch_indexes

def downsampling_data(df, targets, goal_percentages=None, positive_oversampling=True):
    """ Downsample data to achieve or at least approximate to the goal percentage """
    if goal_percentages is None:
        return df

    if not positive_oversampling:
        pass

    total_rows = np.empty(len(targets))
    for i in range(len(targets)):
        condition = df[targets[i]] == 1
        total_rows[i] = len(df[condition].index) // goal_percentages[i]

    i_min = np.argmin(total_rows)

    # Choose between:
    # 1) Use previous conditions to not allow targets which have been already used (exact method)
    # 2) Store indexes already used to not duplicate (aproximate method)
    prev_conditions = df[targets[i_min]] == 1
    used_index = np.array([])

    sort_totals = dict(sorted(zip(total_rows, range(len(total_rows)))))
    totals_dict = dict([(value, key) for key, value in sort_totals.items()])

    first_loop = True
    for i, total in totals_dict.items():
        if i == i_min:
            continue

        print(f"Treating target {i}")
        taget_cond = df[targets[i]] == 1

        num_samples = int(total_rows[i_min] * goal_percentages[i])

        if first_loop:
            num_samples_yet = 0
            downsampling_df = df[taget_cond & ~prev_conditions].copy()
            first_loop = False
        else:
            num_samples_yet = len(downsampling_df[taget_cond])
            num_samples_add = num_samples - num_samples_yet

            if num_samples_add > 0:
                if num_samples_add > len(df[taget_cond & ~prev_conditions].index):
                    print(
                        f"Rows added from target {i} are {len(df[taget_cond & ~prev_conditions].index)}"
                    )
                    downsampling_df = pd.concat(
                        [downsampling_df, df[taget_cond & ~prev_conditions]]
                    )
                else:
                    print(f"Rows added from target {i} are {num_samples_add}")
                    downsampling_df = pd.concat(
                        [
                            downsampling_df,
                            df[taget_cond & ~prev_conditions].sample(
                                num_samples_add, random_state=42
                            ),
                        ]
                    )
            else:
                downsampling_df = downsampling_df.drop(
                    downsampling_df[taget_cond].sample(-num_samples_add).index
                )
                print(f"No Rows added from target {i}")

        prev_conditions = prev_conditions | taget_cond

    num_samples_add = int(
        total_rows[i_min]
        - len(downsampling_df.index)
        - int(total_rows[i_min] * goal_percentages[i_min])
    )
    print(f"Rows with no positive target {num_samples_add}")

    if num_samples_add > 0:
        if num_samples_add > len(df[~prev_conditions].index):
            downsampling_df = pd.concat([downsampling_df, df[~prev_conditions]])
        else:
            downsampling_df = pd.concat(
                [
                    downsampling_df,
                    df[~prev_conditions].sample(num_samples_add, random_state=42),
                ]
            )

    return pd.concat([df[df[targets[i_min]] == 1], downsampling_df]).sort_index()
        
def oversampling_data(
    df, target, goal_percentage=1 / 3, positive_oversampling=True, max_ratio=10
):
    """ Oversample data to achieve or at least approximate to the goal percentage """
    current_percentage = df[target].mean()
    current_percentage = (
        current_percentage if positive_oversampling else 1 - current_percentage
    )
    if goal_percentage <= current_percentage:
        return df
    else:
        max_rows = int(np.round(goal_percentage * len(df)))
        current_rows = int(np.round(current_percentage * len(df)))
        ratio = int(np.ceil(goal_percentage // current_percentage) - 1)

        exceeds_max_ratio = ratio > max_ratio
        if exceeds_max_ratio:
            ratio = max_ratio

        ratio = max_ratio if ratio > max_ratio else ratio
        over_df = df[df[target] == positive_oversampling]
        i = 0
        while i < ratio:
            df = pd.concat([df, over_df])
            i = i + 1

        if not exceeds_max_ratio and current_rows * (1 + ratio) < max_rows:
            over_df = df[df[target] == positive_oversampling].sample(
                max_rows - current_rows * (1 + ratio), replace=True
            )
            df = pd.concat([df, over_df])

        # Shuffle and reindex dataframe
        return df.sample(frac=1).reset_index(drop=True)

def convert_Int_to_int(df, columns=None, verbose=True):
    """ Convert Pandas Int dtype to corresponding numpy int dtype """
    if columns == None:
        columns = list(df.columns)
    df_int_cols = list(df[columns].dtypes[df.dtypes.astype(str).str.lower().str.startswith('int')].index)
    df_int_cols_not_null = df[df_int_cols].dtypes[df[df_int_cols].notnull().any()]

    numerics = {'Int8': np.int8, 'Int16': np.int16, 'Int32': np.int32, 'Int64': np.int64}

    for col_type, np_type in tqdm_notebook(numerics.items(), disable=not verbose):
        cols_to_convert = list(df_int_cols_not_null[df_int_cols_not_null == col_type].index)

        if len(cols_to_convert) > 0:
            df[cols_to_convert] = df[cols_to_convert].astype(np_type)

    return df

def KFolds_stratified(dataframe, k=10, target="class", shuffle=True, seed=None):
    """ Generate kFolds pairs of training and validation sets """
    train_folds = []
    valid_folds = []
    Kfolds = []

    stratify_df = dataframe[target] if shuffle else None
    remain_df, kfold_df = train_test_split(
        dataframe,
        test_size=1 / (k),
        stratify=stratify_df,
        shuffle=shuffle,
        random_state=seed,
    )

    train_folds.append(kfold_df)

    i = 1
    while i < k - 1:
        try:
            stratify_df = remain_df[target] if shuffle else None
            remain_df, kfold_df = train_test_split(
                remain_df,
                test_size=1 / (k - i),
                stratify=stratify_df,
                shuffle=shuffle,
                random_state=seed,
            )
        except ValueError as e:
            print(f"Stratify is not posible at kfold {i} due to: {e}")
            remain_df, kfold_df = train_test_split(
                remain_df, test_sie=1 / (k - i), shuffle=shuffle, random_state=seed
            )

        train_folds.append(kfold_df)
        valid_folds.append(kfold_df)

        i = i + 1

    valid_folds.append(remain_df)
    i = 0
    while i < k - 1:
        Kfolds.append((train_folds[i], valid_folds[i]))
        i = i + 1

    return Kfolds

def reduce_mem_usage(df, columns=None, verbose=True, debug=False):
    """ Reduce memory usage of provided DataFrame using best dtype for each column """
    if columns == None:
        columns = df.columns
    elif len(columns) == 0:
        return df

    if verbose:
        print("Starting reducing memory usage of provided DataFrame")
        if debug:
            print("Identifying integers and float columns")

    df_float_cols = list(df[columns].dtypes[df.dtypes.astype(str).str.startswith('float')].index)

    float_to_int_cols = np.equal(df[df_float_cols].fillna(0) - df[df_float_cols].fillna(0).apply(np.floor), 0).all()
    float_to_int_cols = list(df[df_float_cols].dtypes[float_to_int_cols].index)
    df_float_cols = list(set(df_float_cols) - set(float_to_int_cols))

    df_int_cols = list(
        df[columns].dtypes[df.dtypes.astype(str).str.lower().str.startswith('int')].index) + float_to_int_cols
    df_int_cols_not_null = df[df_int_cols].dtypes[df[df_int_cols].notnull().any()]
    df_cols = df_float_cols + df_int_cols

    numerics = {'Int8': np.int8, 'Int16': np.int16, 'Int32': np.int32, 'Int64': np.int64, 'float32': np.float32,
                'float64': np.float64}

    df_types = pd.DataFrame(0, index=df_cols, columns=list(numerics.keys()))
    df_types_min_max = pd.DataFrame(0, index=['min', 'max'], columns=list(numerics.keys()))
    df_min_max = pd.DataFrame(0, index=['min', 'max'], columns=df_cols)

    for col_type, np_type in numerics.items():
        if str(col_type)[:3].lower() == 'int':
            df_types_min_max.loc['min', col_type] = np.iinfo(np_type).min
            df_types_min_max.loc['max', col_type] = np.iinfo(np_type).max
        else:
            df_types_min_max.loc['min', col_type] = np.finfo(np_type).min
            df_types_min_max.loc['max', col_type] = np.finfo(np_type).max

    if debug:
        print("Collecting min and max values of all columns")

    df_min_max.loc['min'] = df[df_cols].min(skipna=True).T
    df_min_max.loc['max'] = df[df_cols].max(skipna=True).T

    df_max = pd.DataFrame(0, index=df_cols, columns=list(numerics.keys()))
    df_max.loc[df_cols, list(numerics.keys())] = df_min_max.loc['max', df_cols]
    df_max = df_max < df_types_min_max.loc['max', list(numerics.keys())]

    df_min = pd.DataFrame(0, index=df_cols, columns=list(numerics.keys()))
    df_min.loc[df_cols, list(numerics.keys())] = df_min_max.loc['min', df_cols]
    df_min = df_min > df_types_min_max.loc['min', list(numerics.keys())]

    df_types = df_max & df_min
    df_types.loc[df_float_cols, ['Int8', 'Int16', 'Int32', 'Int64']] = False
    df_types.loc[df_int_cols, ['float32', 'float64']] = False

    if debug:
        print("Collecting current types of each column")

    df_current_types = pd.DataFrame(False, index=df_cols, columns=list(numerics.keys()))
    for col_type in numerics.keys():
        df_current_types.loc[list(df[df_cols].dtypes[df[df_cols].dtypes == col_type].index), col_type] = True

    if debug:
        print("Calculating initial memory usage")

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    cols_converted = []

    if debug:
        print("Starting converting columns to the optimal types")

    for col_type, np_type in tqdm_notebook(numerics.items(), disable=not verbose):
        cols_able_to_convert = list(df_types[col_type][df_types[col_type]].index)
        cols_current_type = list(df_current_types.loc[df_current_types[col_type], col_type].index)
        cols_to_convert = list(set(cols_able_to_convert) - set(cols_converted) - set(cols_current_type))
        cols_not_changed = [col for col in cols_able_to_convert if col in cols_current_type]

        if col_type[:3] == 'Int':
            cols_to_convert_int_not_null = [col for col in cols_to_convert if col in df_int_cols_not_null]

            if len(cols_to_convert_int_not_null) > 0:
                df[cols_to_convert_int_not_null] = df[cols_to_convert_int_not_null].astype(np_type)

                cols_to_convert = list(set(cols_to_convert) - set(cols_to_convert_int_not_null))
        
        if len(cols_to_convert) > 0:
            if verbose:
                print("Converting to ", col_type, " ", len(cols_to_convert), " columns")

            df[cols_to_convert] = df[cols_to_convert].astype(col_type)

        cols_converted = cols_converted + cols_to_convert + cols_not_changed

    if debug:
        print("Calculating final memory usage")
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))

    return df