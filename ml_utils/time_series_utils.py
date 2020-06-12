import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

from ml_utils.plot_utils import plot_scatter


def laggingData(X, data, lag, suffix='Prev_', data_cols=None, exclude_cols=['datetime'], output_data=False):
    """ Generate new X containing lagged data """
    if data_cols is None:
        data_cols = [col for col in data.columns if col not in exclude_cols]

    # Set temporal Y which will be added to shifted X
    temp_data = data[data_cols].copy()
    temp_data.columns = [suffix + col for col in data_cols]
    cols_created = list(temp_data.columns)

    # Shift X to match the temporal Y and remove NaNs from end of X
    new_X = pd.concat([X.shift(-lag), temp_data], axis=1, sort=False)
    new_X = new_X.iloc[0: -lag].reset_index(drop=True)

    # Create the new Y which does not contain the info added to X
    if output_data:
        new_data = data.iloc[lag:].reset_index(drop=True)

        return new_X, new_data, cols_created
    else:
        return new_X, cols_created


def dataToSequential(data, seq_len, out_np=True, exclude_cols=['datetime'], dtype=np.float32):
    """ Transform data to sequential data """
    new_data = data.copy()
    data_cols = [col for col in data.columns if col not in exclude_cols]
    for i in tqdm_notebook(range(1, seq_len + 1)):
        columns = ['Prev_' + str(i) + '_' + col for col in data_cols]
        new_data[columns] = data[data_cols].shift(i).reset_index(drop=True)

    new_data = new_data.iloc[seq_len:].reset_index(drop=True)

    if out_np:
        return pdToNpSequential(new_data, seq_len, dtype)

    return new_data


def pdToNpSequential(data, seq_len, dtype=np.float32):
    """ Transform from pandas 2D (rows, cols * sequence_length) to numpy 3D (rows, sequence_length, cols)"""
    return np.array(data, dtype=dtype).reshape(data.shape[0], seq_len + 1, data.shape[1] // (seq_len + 1))


# Input data for DNN should be numpy 3D data and it must no contain 'datetime' column
def prepareInputData(data, seq_len, keep_datetime, dtype=np.float32):
    """ Transform data to be used as input on DNN """
    if keep_datetime:
        return pdToNpSequential(data.drop('datetime', axis=1), seq_len, dtype=dtype)

    return pdToNpSequential(data, seq_len, dtype=dtype)


def medial_average(x):
    """ Calculate medial average which is average excluding the maximum and minimum value  """
    return x.loc[x.notnull()].sort_values().iloc[1:-1].mean()


def seasonal_mean(x, freq, window_length=24, agg_type='mean'):
    """ Calculate seasonal mean or seasonal median with the provided frequency and window length """
    temp_df = pd.DataFrame(x, index=x.index, columns=['value'])
    if freq == 1:
        tile_arr = np.arange(window_length)
    else:
        tile_arr = np.repeat(np.arange(freq), window_length)

    # Set same values for the points which should be aggregated
    temp_df['window_point'] = np.tile(tile_arr, len(x.index) // (window_length * freq) + 1)[:len(x.index)]
    if agg_type == 'mean':
        agg_func = np.mean
    elif agg_type == 'median':
        agg_func = np.median
    else:
        raise NotImplementedError("agg_type different than mean or median have not been implemented")
    return temp_df.groupby(['window_point'])['value'].transform(agg_func)


def extract_seasonal_component(df, columns_to_treat, daily_windows=[], agg_type='mean', win_type='boxcar',
                               each_row_is='hour', plot=False):
    """ Calculate Seasonal component  (additive and multiplicative) and trend for the given columns and for each daily window """
    if each_row_is != 'hour':
        raise NotImplementedError

    cols_created = []
    temp_df = df.copy()
    for col in columns_to_treat:

        # Calculate number of non-null values on each day for the current column
        daily_nonull_window = df[col].iloc[0:24].notnull().sum()

        temp_df['adjusted_add_' + col] = temp_df[col].copy()
        temp_df['adjusted_mult_' + col] = temp_df[col].copy()
        for days in sorted(daily_windows, reverse=True):

            # As the window only takes non-null values, we have to set the window to only take the exact number of days
            rolling_window = daily_nonull_window * days
            if win_type == 'exponential':
                rolling_window = (rolling_window, days / 2)
            else:
                if agg_type == 'median':
                    win_type = None

            # Trend is the result of the moving average over the rolling window
            temp_df.loc[temp_df['adjusted_add_' + col].notnull(), str(days) + '_trend_' + col] = temp_df.loc[
                temp_df['adjusted_add_' + col].notnull(), 'adjusted_add_' + col].rolling(window=rolling_window,
                                                                                         win_type=win_type).agg(
                agg_type)

            # Seasonal component is the result of the seasonal_mean on the time series substracted (or divided) by trend
            temp_df[str(days) + '_seasonal_comp_add_' + col] = seasonal_mean(
                temp_df['adjusted_add_' + col] - temp_df[str(days) + '_trend_' + col], days, agg_type=agg_type)
            temp_df[str(days) + '_seasonal_comp_mult_' + col] = seasonal_mean(
                temp_df['adjusted_mult_' + col] / temp_df[str(days) + '_trend_' + col], days, agg_type=agg_type)

            # Adjusted serie is achieve substracting (or dividing) the data serie by the seasonal component
            temp_df['adjusted_add_' + col] = temp_df['adjusted_add_' + col] - temp_df[
                str(days) + '_seasonal_comp_add_' + col]
            temp_df['adjusted_mult_' + col] = temp_df['adjusted_mult_' + col] / temp_df[
                str(days) + '_seasonal_comp_mult_' + col]

            cols_created.append(str(days) + '_trend_' + col)
            cols_created.append(str(days) + '_seasonal_comp_add_' + col)
            cols_created.append(str(days) + '_seasonal_comp_mult_' + col)

        cols_created.append('adjusted_add_' + col)
        cols_created.append('adjusted_mult_' + col)

    if plot:
        nrow = 0
        # Resize for better visualization of subplots
        subplot_rows = (len(daily_windows) + 1) * len(columns_to_treat)
        subplot_cols = 3
        plt.rcParams['figure.figsize'] = [subplot_cols * 5, subplot_rows * 4]

        fig, axes = plt.subplots(subplot_rows, subplot_cols, sharex=False, sharey=False)

        for col in columns_to_treat:
            plot_scatter(temp_df, 'datetime', col, axes=axes[nrow, 0])
            plot_scatter(temp_df, 'datetime', 'adjusted_add_' + col, axes=axes[nrow, 1])
            plot_scatter(temp_df, 'datetime', 'adjusted_mult_' + col, axes=axes[nrow, 2])
            nrow += 1
            for days in sorted(daily_windows, reverse=True):
                if days == 1:
                    freq = 1
                else:
                    freq = 24
                plot_scatter(temp_df, 'datetime', str(days) + '_trend_' + col, axes=axes[nrow, 0])
                plot_scatter(temp_df.iloc[:days * 24:freq], 'datetime', str(days) + '_seasonal_comp_add_' + col,
                             axes=axes[nrow, 1])
                plot_scatter(temp_df.iloc[:days * 24:freq], 'datetime', str(days) + '_seasonal_comp_mult_' + col,
                             axes=axes[nrow, 2])
                nrow += 1

        # Resize to original settings
        plt.rcParams['figure.figsize'] = [10, 6]

    return temp_df[cols_created], cols_created
