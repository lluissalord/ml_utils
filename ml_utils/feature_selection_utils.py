import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import os

from ml_utils.plot_utils import plot_scatter, get_subplot_rows_cols

def covariate_shift(train, test, categorical_columns, n_samples, iterations = 200, weights_coef = 1, AUC_threshold = 0.8, importance_threshold = 0.9, max_loops = 20, test_size = 0.1, trys_all_influencer=5, calc_sample_weights=True, task_type="CPU", data_dir='', load_cov=False, save_cov=False, plot=True):
    """ Select features without Covariate Shift between training and test set using iteratively CatBoostClassifier to identify relation between train and test """
    
    import seaborn as sns
    import catboost as cb
    from sklearn.model_selection import train_test_split
    
    if not os.path.exists(data_dir + 'cov_shift_features.pkl') or not load_cov:
        train_sample = train.sample(n_samples)
        train_sample.loc[:,'origin'] = 0

        test_sample = test.sample(n_samples)
        test_sample.loc[:,'origin'] = 1

        combined_train, combined_test = train_test_split(
            pd.concat([train_sample.reset_index(drop=True), test_sample.reset_index(drop=True)]), 
            test_size = test_size, 
            shuffle = True)

        try:

            influence_columns = []
            count_all_influencer = 0
            i = 0
            AUC_score = 1
            while i < max_loops and AUC_score > AUC_threshold:

                x_columns = combined_train.columns.drop(['origin',] + influence_columns)

                # Get the indexes for the categorical columns which CatBoost requires to out-perform other algorithms
                cat_features_index = [list(x_columns).index(col) for col in categorical_columns if col in list(x_columns)]

                # Do the feature selection once and only try again if no feature is selected
                cov_shift_feature_selection = []
                while len(cov_shift_feature_selection) == 0 and count_all_influencer < trys_all_influencer:
                    if count_all_influencer > 0:
                        print("Try again because model has set any feature as influencer")
                    
                    cov_shift_model = cb.CatBoostClassifier(iterations = iterations,
                                                            eval_metric = "AUC",
                                                            cat_features = cat_features_index,
                                                            task_type = task_type,
                                                            verbose = False
                                                   )
                    cov_shift_feature_selection, df_cov_shift_feature_selection = shadow_feature_selection(
                        cov_shift_model, 
                        combined_train['origin'], combined_train[x_columns], 
                        need_cat_features_index=True, categorical_columns=categorical_columns, 
                        collinear_threshold = 1,
                        n_iterations_mean = 1, times_no_change_features = 1
                    )
                    
                    count_all_influencer += 1
                
                if count_all_influencer == trys_all_influencer:
                    cov_shift_feature_selection = list(x_columns)
                
                # Get the indexes for the categorical columns which CatBoost requires to out-perform other algorithms
                cat_features_index = [cov_shift_feature_selection.index(col) for col in categorical_columns if col in cov_shift_feature_selection]

                params = {'iterations' : 2*iterations, 'learning_rate' : 0.05, 'depth' : 6}

                cov_shift_model = cb.CatBoostClassifier(iterations = iterations,
                                                   eval_metric = "AUC",
                                                   cat_features = cat_features_index,
                                                   scale_pos_weight = combined_train['origin'].value_counts()[0] / combined_train['origin'].value_counts()[1],
                                                   task_type = task_type,
                                                   verbose = False
                                               )

                cov_shift_model.set_params(**params)
                cov_shift_model.fit(combined_train.drop('origin', axis = 1)[cov_shift_feature_selection],
                                   combined_train['origin'], 
                                   eval_set = (combined_test.drop('origin', axis = 1)[cov_shift_feature_selection], combined_test['origin']),
                                   use_best_model = True,
                                   #sample_weight = sample_weight,
                                   #early_stopping_rounds = True,
                                   plot = False,
                                   verbose = False)

                AUC_score = cov_shift_model.get_best_score()['validation']['AUC']
                print(f"Model score AUC of {AUC_score} on test")

                # Remove the features which cumulative importance is relevant to predict origin of data (train or test)
                if count_all_influencer != trys_all_influencer:
                    df_cov_shift_importance = pd.DataFrame(cov_shift_model.feature_importances_, columns = ['importance'], index = cov_shift_feature_selection)
                    df_cov_shift_importance['cumulative_importance'] = df_cov_shift_importance['importance'].cumsum() / df_cov_shift_importance['importance'].sum()

                    new_influence_columns = list(df_cov_shift_importance[df_cov_shift_importance['cumulative_importance'] < importance_threshold].index)
                    influence_columns = influence_columns + new_influence_columns

                    print(f"New {len(new_influence_columns)} columns will be removed from model: ", new_influence_columns)
                    print()
                    
                count_all_influencer = 0
                
                i = i + 1
        finally:

            print()
            print(f"Due to difference of influence of features to distinguish between data and submission, {len(influence_columns)} columns are removed:")
            print(influence_columns)
            
            if calc_sample_weights:
                print("Calculating weights for each training sample")
                probs = cov_shift_model.predict_proba(train[cov_shift_model.feature_names_])[:, 1] #calculating the probability

                #print("Plot Train AUC")
                #plot_roc_auc(pd.Serie(1,index = train.index), probs)

                sample_weight = -np.log(probs) 
                sample_weight /= max(sample_weight) # Normalizing the weights

                sample_weight = 1 + weights_coef * sample_weight

                if plot:
                    plt.xlabel('Computed sample weight')
                    plt.ylabel('# Samples')
                    sns.distplot(sample_weight, kde=False)
            
            if save_cov:
                with open(data_dir + 'cov_shift_features.pkl', 'wb') as file:
                    print("Saving data in ", data_dir + 'cov_shift_features.pkl')
                    pickle.dump(influence_columns, file)
    else:
        print("Loading influence columns from ",data_dir)

        with open(data_dir + 'cov_shift_features.pkl', 'rb') as file:
            influence_columns = pickle.load(file)

        cov_shift_model = None
        sample_weight = [1,] * len(train)
            
    return influence_columns, cov_shift_model, sample_weight

def stadistic_difference_distributions(data, submission, time_column, test_percentage=0.2, p_value_threshold=None,
                                       verbose=False):
    """ Calculate relation between initial and end part of the dataset for each column using Kolmogorov-Smirnov statistic on 2 samples """
    
    from scipy import stats
    from sklearn.model_selection import train_test_split
    
    train, test = train_test_split(data.sort_values(time_column), test_size=test_percentage, shuffle=False)

    time_analysis_df = pd.DataFrame(False, columns=['train_test', 'train_submission', 'test_submission'],
                                    index=submission.columns.values)

    for col in tqdm_notebook(submission.columns.values):
        try:
            KS_stat_test, p_value_test = stats.ks_2samp(train[col], test[col])
            KS_stat_submission, p_value_submission = stats.ks_2samp(train[col], submission[col])
            KS_stat_test_submission, p_value_test_submission = stats.ks_2samp(test[col], submission[col])

            time_analysis_df.loc[col] = [p_value_test, p_value_submission, p_value_test_submission]

            if verbose:
                if p_value_test <= p_value_threshold or p_value_submission <= p_value_threshold or p_value_test_submission <= p_value_threshold:
                    print_s = f'Column {col} has different distribution'
                    if p_value_test <= p_value_threshold:
                        print_s = print_s + ' // train <--> test'
                    if p_value_submission <= p_value_threshold:
                        print_s = print_s + ' // train <--> submission'
                    if p_value_test_submission <= p_value_threshold:
                        print_s = print_s + ' // test <--> submission'
                    print(print_s)
        except TypeError:
            time_analysis_df.loc[col] = [np.nan, np.nan, np.nan]

    if p_value_threshold == None:
        cond1 = time_analysis_df['train_test'] == 0
        cond2 = time_analysis_df['train_submission'] == 0
        cond3 = time_analysis_df['test_submission'] == 0
    else:
        cond1 = time_analysis_df['train_test'] <= p_value_threshold
        cond2 = time_analysis_df['train_submission'] <= p_value_threshold
        cond3 = time_analysis_df['test_submission'] <= p_value_threshold

    cols_to_remove = list(time_analysis_df[cond1 | cond2 | cond3].index)

    return time_analysis_df, cols_to_remove

def outliers_analysis(full_data, features_names=None, x_column=None, subplot_rows=None, subplot_cols=None, starting_index=0,
                      index_offset=0, z_score_threshold=3.5, use_mean=False, plot=True, num_bins=50):
    """ Calculate and visualize outliers analysis from Modified Z-score with MAD """
    # Compatibility with numpy arrays
    if type(full_data) == np.ndarray:
        assert len(full_data.shape) <= 2
        if len(full_data.shape) == 1:
            columns = ['feature']
        else:
            columns = ['feature_'+str(i) for i in range(full_data.shape[-1])]
        full_data = pd.DataFrame(full_data, columns=columns)
    
    # Features not provided, use all the columns
    if features_names is None:
        features_names = list(full_data.columns)
    
    if plot:
        # Set a good relation rows/cols for the plot if not specified
        if subplot_rows is None or subplot_cols is None:
            subplot_rows, subplot_cols = get_subplot_rows_cols(len(features_names), [3,4,5])
                    
        # Resize for better visualization of subplots
        plt.rcParams['figure.figsize'] = [subplot_cols * 5, subplot_rows * 4]
        fig, axes = plt.subplots(subplot_rows, subplot_cols, sharex=False, sharey=False)

    outliers_pd = full_data.copy()

    outliers_summary = {}

    i = starting_index
    while i < len(features_names):
        feature_name = features_names[i]

        data = outliers_pd.loc[outliers_pd[feature_name].notnull(), feature_name]

        # Modified Z-score with MAD (Median Absolute Deviation)
        if use_mean:
            outliers_pd.loc[outliers_pd[feature_name].notnull(), feature_name + '_zscore'] = 0.6745 * (data - data.mean()).abs() / (
                data - data.mean()).abs().mean()
        else:
            outliers_pd.loc[outliers_pd[feature_name].notnull(), feature_name + '_zscore'] = 0.6745 * (data - data.median()).abs() / (
                data - data.median()).abs().median()
        outliers_pd[feature_name + '_zscore_outliers'] = outliers_pd[feature_name + '_zscore'] > z_score_threshold

        if plot:
            # Take into account the case of only one plot
            if subplot_rows * subplot_cols == 1:
                ax = axes
            elif subplot_rows == 1:
                ax = axes[(i + index_offset) % subplot_cols]
            else:
                ax = axes[(i + index_offset) // subplot_cols, (i + index_offset) % subplot_cols]
            
            # If X_column provided plot scatter, otherwise histogram
            if x_column is None:
                bins = np.linspace(data.min(), data.max(), num_bins)
                ax.hist(data[~outliers_pd[feature_name + '_zscore_outliers']], bins=bins, density=False)
                ax.hist(data[outliers_pd[feature_name + '_zscore_outliers']], bins=bins, density=False)
                ax.set_title(feature_name)
            else:
                plot_scatter(outliers_pd[outliers_pd[feature_name].notnull()], x_column=x_column, y_column=feature_name,
                         axes=ax, highlight_column=feature_name + '_zscore_outliers')

        outliers_percentage = 100 * outliers_pd[feature_name + '_zscore_outliers'].sum() / outliers_pd[
            feature_name + '_zscore_outliers'].count()
        outliers_summary[feature_name] = outliers_percentage

        print("Feature: ", feature_name, " - Percentage of outliers using modified Z-score approach is: ",
              np.round(outliers_percentage, 2), "%")

        i = i + 1

    if plot:
        fig.tight_layout()

        # Resize to original settings
        plt.rcParams['figure.figsize'] = [10, 6]

    outliers_summary = pd.DataFrame.from_dict(outliers_summary, orient='index', columns=['Percentage'])
        
    return outliers_summary, outliers_pd

def feature_selection(classifier_initial, y_train, x_train, n_top_features=50, baseline_features=[],
                      min_importance=None):
    """ Select features which have the top N feature importance and/or above baseline """
    classifier_model = classifier_initial.fit(x_train, y_train)

    feature_importance = sorted(zip(map(lambda x: round(x, 4), classifier_model.feature_importances_), x_train),
                                reverse=True)
    dict_feature_importance = dict(zip(x_train, map(lambda x: round(x, 4), estimator.feature_importances_)))

    if baseline_features:
        min_importance = max([importance for importance, feature in feature_importance if feature in baseline_features])

    model_columns = []
    i = 0
    while i < n_top_features and i < len(feature_importance):
        if feature_importance[i][0] > min_importance:
            model_columns.append(feature_importance[i][1])
        else:
            break
        i = i + 1

    return model_columns


def cumulative_feature_selection(df_feature_importance, cum_importance_threshold):
    """ Select features which are below of the cumulative feature importance threshold """
    df_feature_importance = pd.DataFrame(df_feature_importance, columns=['importance'])
    df_feature_importance['cumulative_importance'] = df_feature_importance['importance'].cumsum() / \
                                                     df_feature_importance['importance'].sum()

    print("Removed ", sum(df_feature_importance['cumulative_importance'] >= cum_importance_threshold),
          " features due to low importance:")
    print(
        df_feature_importance[df_feature_importance['cumulative_importance'] >= cum_importance_threshold].index.values)

    df_feature_importance = df_feature_importance[
        df_feature_importance['cumulative_importance'] < cum_importance_threshold]

    return df_feature_importance


def collinear_feature_selection(x_train, df_feature_importance, collinear_threshold=0.98, plot=True):
    """ Select features which have less collinearity below the threshold """
    correlation = x_train[df_feature_importance.index].corr()

    if plot:
        correlation.round(3).style.background_gradient(cmap='coolwarm')

    cond1 = pd.DataFrame(np.triu(np.ones(correlation.shape[0]) - np.eye(correlation.shape[0])),
                         columns=correlation.columns, index=correlation.index) == 1
    corr_final = (correlation > collinear_threshold) & cond1
    corr_final = corr_final.loc[:, corr_final.any()]

    features_remove = []
    columns = corr_final.columns.values
    rows = corr_final.index.values

    for i in tqdm_notebook(range(corr_final.shape[1]), desc='1st Loop'):

        # If a feature is already on the remove list, then it is not needed to check
        if columns[i] in features_remove:
            continue

        j_max = np.where(rows == columns[i])

        for j in tqdm_notebook(range(corr_final.shape[0]), desc='2nd Loop', leave=False):

            if j == j_max:
                break

            # Feature columns[i] and feature rows[j] are collinear
            if corr_final.iloc[j, i]:

                # If a feature is already on the remove list, then it is not needed to check
                if rows[j] in features_remove:
                    continue

                # Remove the one which has less importance
                importance_i = df_feature_importance.loc[columns[i], 'importance']
                importance_j = df_feature_importance.loc[rows[j], 'importance']
                if importance_i < importance_j:
                    features_remove.append(columns[i])
                else:
                    features_remove.append(columns[j])

    print("Removed ", len(features_remove), " features due to collinearity: ")
    print(features_remove)
    df_feature_importance = df_feature_importance.drop(features_remove)

    return df_feature_importance


def shadow_feature_selection(classifier_initial, y_train, x_train, eval_set=None, n_top_features=None, collinear_threshold=0.98, cum_importance_threshold=0.99, times_no_change_features=2, max_loops=50, n_iterations_mean=3, need_cat_features_index=False, categorical_columns=[], verbose=True, debug=False, plot_correlation=False):
    """ Select features which have greater feature importance than random and than shuffle shadowed feature of itself """
    # Create 3 random features which will serve as baseline to reject features
    baseline_features = ['random_binary', 'random_uniform', 'random_integers']
    x_train = x_train.drop(baseline_features, axis=1, errors='ignore')
    x_train['random_binary'] = np.random.choice([0, 1], x_train.shape[0])
    x_train['random_uniform'] = np.random.uniform(0, 1, x_train.shape[0])
    x_train['random_integers'] = np.random.randint(0, x_train.shape[0] / 2, x_train.shape[0])

    # For each feature it creates a shadow_feature which will have same values but shuffled
    x_train_shadow, dict_shadow_names = _create_shadow(x_train, baseline_features)

    count_no_changes = 0
    x_all_columns = baseline_features + list(dict_shadow_names.keys()) + list(dict_shadow_names.values())

    # "Infinite" loop till one of the stopping criterias stop removing features
    for i in tqdm_notebook(range(max_loops), desc='Main Loop'):
        if dict_shadow_names == {}:
            print("Stopping because no feature found to be relevant")
            return [], pd.DataFrame([])
        
        print("Loop number: ", i, " with still ", len(dict_shadow_names.keys()), " features")

        # Take a copy of current columns to check stopping criteria of changing columns
        x_all_columns_prev = x_all_columns.copy()
        x_all_columns_prev.sort()
        if debug:
            print("x_all_columns_prev: ", x_all_columns_prev)

        # Get the feature importance for each column (real, shadow and baseline)
        df_feature_importance = get_feature_importance_mean(classifier_initial, y_train, x_train_shadow[x_all_columns],
                                                            n_iterations_mean=n_iterations_mean,
                                                            need_cat_features_index=need_cat_features_index,
                                                            categorical_columns=categorical_columns,
                                                            dict_shadow_names=dict_shadow_names,
                                                            eval_set=eval_set)

        # Take as minimum value of feature importance as the greatest value of the baselines features
        if baseline_features != []:
            min_importance = df_feature_importance[baseline_features].max()
        else:
            min_importance = 0

        # Drop all features that have lower importance than their shadow or lower than the baseline features
        dict_shadow_names_copy = dict_shadow_names.copy()
        for real_col, shadow_col in dict_shadow_names.items():
            if df_feature_importance[shadow_col] >= df_feature_importance[real_col] or df_feature_importance[
                real_col] < min_importance:
                del dict_shadow_names_copy[real_col]

                if debug:
                    print("Removing feature: ", real_col)

        if debug:
            print("dict_shadow_names: ", dict_shadow_names)
            print("--------------------------------------------------------")
            print("dict_shadow_names_copy: ", dict_shadow_names_copy)
            print()
            print("--------------------------------------------------------")
            print()
        dict_shadow_names = dict_shadow_names_copy.copy()

        if debug:
            print("dict_shadow_names: ", dict_shadow_names)
            print("--------------------------------------------------------")
            print("dict_shadow_names_copy: ", dict_shadow_names_copy)

        x_all_columns = baseline_features + list(dict_shadow_names.keys()) + list(dict_shadow_names.values())

        x_all_columns.sort()
        if debug:
            print()
            print("--------------------------------------------------------")
            print()
            print("x_all_columns_prev: ", x_all_columns_prev)
            print("--------------------------------------------------------")
            print("x_all_columns: ", x_all_columns)

        # Check if has been any change, if not then break the loop
        if x_all_columns == x_all_columns_prev:
            count_no_changes += 1
            if times_no_change_features == count_no_changes:
                print("Stopping feature selection due to no change")
                break
        else:
            count_no_changes = 0
            print("This loop has removed ", int((len(x_all_columns_prev) - len(x_all_columns)) / 2), " features")

        # If we have reduced to the maximum number of features, then break the loop
        if n_top_features != None and n_top_features >= len(dict_shadow_names.keys()):
            print("Stopping feature selection due to reached maximum of features")
            break

    df_feature_importance = df_feature_importance[list(dict_shadow_names.keys())].sort_values(ascending=False)

    df_feature_importance = cumulative_feature_selection(df_feature_importance, cum_importance_threshold)

    df_feature_importance = collinear_feature_selection(x_train[df_feature_importance.index.values],
                                                        df_feature_importance, collinear_threshold=collinear_threshold,
                                                        plot=plot_correlation)

    if n_top_features != None:
        if n_top_features >= len(dict_shadow_names.keys()):
            df_feature_importance = df_feature_importance.iloc[0: len(dict_shadow_names.keys())]
        else:
            df_feature_importance = df_feature_importance.iloc[0: n_top_features]

    return list(df_feature_importance.index), df_feature_importance


def _create_shadow(x, baseline_features):
    """
    Take all X variables, creating copies and randomly shuffling them
    :param x: the dataframe to create shadow features on
    :return: dataframe 2x width and the names of the shadows for removing later
    """
    print("Creating all shadow features")
    x_shadow = x.copy()
    x_shadow = x_shadow.drop(baseline_features, axis=1)
    for c in tqdm_notebook(x_shadow.columns, desc='Shadow Cols'):
        np.random.shuffle(x_shadow[c].values)  # shuffle the values of each feature to all the features
    # rename the shadow
    shadow_names = ["shadow_feature_" + str(i + 1) for i in range(x_shadow.shape[1])]
    dict_shadow_names = dict(zip(x_shadow.columns.copy(), shadow_names))
    x_shadow.columns = shadow_names
    # Combine to make one new dataframe
    x_new = pd.concat([x, x_shadow], axis=1)
    return x_new, dict_shadow_names


def get_feature_importance_mean(classifier_initial, y_train, x_train, n_iterations_mean=3,
                                need_cat_features_index=False, categorical_columns=[], dict_shadow_names={}, eval_set=None):
    """ Calculate feature importance mean between different training of the classifier """
    cat_features_index = None
    if need_cat_features_index:
        if dict_shadow_names=={}:
            raise ValueError("dict_shadow_names cannot be empty if categorical features index are needed")
        
        x_columns = list(x_train.columns)
        cat_features_index = []
        for col in categorical_columns:
            if col in x_columns:
                cat_features_index.extend((x_columns.index(col), x_columns.index(dict_shadow_names[col])))

    n_targets = y_train.shape[-1] if len(y_train.shape) > 1 else 1
    for t in tqdm_notebook(range(n_targets), desc='Target Loop', leave=False):
        for i in tqdm_notebook(range(n_iterations_mean), desc='Mean Loop', leave=False):
            classifier = classifier_initial.copy()
            classifier.set_params(cat_features=cat_features_index, random_state=np.random.randint(100))
            metric_period = int(classifier.get_param('iterations')) // 10 
            if n_targets > 1:
                classifier = classifier.fit(x_train, y_train.iloc[:,t], eval_set=eval_set, metric_period=metric_period)
            else:
                classifier = classifier.fit(x_train, y_train, eval_set=eval_set, metric_period=metric_period)

            feature_importance = sorted(zip(map(lambda x: round(x, 4), classifier.feature_importances_), x_train),
                                        reverse=True)

            if t==0 and i == 0:
                df_feature_importance = pd.DataFrame(dict(zip(x_train, classifier.feature_importances_)), index=[i])
            else:
                df_feature_importance_iter = pd.DataFrame(dict(zip(x_train, classifier.feature_importances_)),
                                                          index=[i])

                df_feature_importance = pd.concat([df_feature_importance, df_feature_importance_iter])

    return df_feature_importance.mean()
