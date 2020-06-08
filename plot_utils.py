import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn import metrics
from tqdm import tqdm_notebook
import seaborn as sns
from scipy.stats import norm

def get_subplot_rows_cols(num_plots, n_cols = [3,4,5]):
    """ Calculate number row and cols for a pretty grid """
    max_cols = max(n_cols)
    if num_plots // max_cols == 0:
        subplot_rows = 1
        subplot_cols = num_plots
    else:
        found = False
        better_res = max(n_cols)
        better_n = None
        for n in sorted(n_cols, reverse=True):
            res = num_plots % n
            if res == 0:
                subplot_rows = num_plots // n
                subplot_cols = n
                found = True
                break
            elif better_res > res:
                better_res = res
                better_n = n

        if not found:
            subplot_rows = num_plots // better_n + 1
            subplot_cols = better_n
    return subplot_rows, subplot_cols

def plot_histograms(df, column_names, df_types, max_value_counts, subplot_rows=None, subplot_cols=None,
                    starting_index=0, index_offset=0, fit=norm):
    """ Plot histogram plot grid for all the feature provided """
    # Set a good relation rows/cols for the plot if not specified
    if subplot_rows is None or subplot_cols is None:
        subplot_rows, subplot_cols = get_subplot_rows_cols(len(column_names), [3,4,5])
    
    # Resize for better visualization of subplots
    plt.rcParams['figure.figsize'] = [subplot_cols * 5, subplot_rows * 4]

    i = starting_index
    while i < len(column_names):
        column_name = column_names[i]
        plt.subplot(subplot_rows, subplot_cols, i + index_offset + 1)

        # Plot using value_counts for integer columns with less than 'max_value_counts' different values,
        # Otherwise, use histogram directly
        if column_name in df_types[df_types != 'float64'].index.tolist() and df[column_name].value_counts().shape[
            0] < max_value_counts:
            df[column_name].value_counts(dropna=False).sort_index().plot(kind='bar', rot=0).set_title(column_name);
        else:
            sns.distplot(df[column_name][df[column_name].notnull()], fit=fit, kde=False)
        
        i = i + 1

    # Resize to original settings
    plt.rcParams['figure.figsize'] = [10, 6]

def plot_roc_auc(y_true, y_pred):
    """ Plot ROC AUC """
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# Settings for axis to plot X data as dates correctly
def plot_date(x, y, axes=None, title=None, color='b'):
    """ Plot data correctly using X date data """
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        
    if type(color) is str:
        axes.plot_date(x, y, color=color)
    elif type(color) is list:
        unique_colors = np.unique(color)
        
        color = np.array(color)
        for unique_color in unique_colors:
            list_unique_color = np.array([unique_color]*len(color))
            cond = np.where(color == list_unique_color, True, False)
            try:
                axes.plot_date(x.loc[cond], y.loc[cond], color=unique_color)
            except AttributeError:
                axes.plot_date(x[cond], y[cond], color=unique_color)
    else:
        raise ValueError("color parameter must be string or list of strings")
        
    if title != None:
        axes.set_title(title)
    
    try:
        datediff = x.iloc[-1] - x.iloc[0]
    except AttributeError:
        datediff = x[-1] - x[0]
    
    if datediff.days >= 365:
        major_loc = mdates.YearLocator()   # every year
        major_format = mdates.DateFormatter('%Y')

        minor_loc = mdates.MonthLocator()  # every month
        axes.xaxis.set_minor_locator(minor_loc)

    elif datediff.days >= 32:
        major_loc = mdates.MonthLocator()   # every month
        major_format = mdates.DateFormatter('%m')

    elif datediff.days >= 7:
        major_loc = mdates.DayLocator(interval=5)   # every day
        major_format = mdates.DateFormatter('%d')

        minor_loc = mdates.DayLocator()
        axes.xaxis.set_minor_locator(minor_loc)

    elif datediff.days >= 1:
        major_loc = mdates.DayLocator()   # every day
        major_format = mdates.DateFormatter('%a')

    else:
        major_loc = mdates.HourLocator(interval=3)   # every day
        major_format = mdates.DateFormatter('%H')

        minor_loc = mdates.HourLocator()
        axes.xaxis.set_minor_locator(minor_loc)

    # format the ticks
    axes.xaxis.set_major_locator(major_loc)
    axes.xaxis.set_major_formatter(major_format)


def plot_scatter(data, x_column, y_column, title=None, axes=None, highlight_column=None, highlight_color='r',
                 normal_color='b'):
    """ Plot scatter plots where we can highlight points as the outliers """

    if highlight_column != None:
        data.loc[data[highlight_column] == True, highlight_column] = highlight_color
        data.loc[data[highlight_column] == False, highlight_column] = normal_color
        color_pd = data[highlight_column]
        # color_pd = data[highlight_column] * highlight_color + (1 - data[highlight_column]) * normal_color
        color_list = color_pd[color_pd.notnull()].values.tolist()
    else:
        color_list = len(data[y_column]) * [normal_color, ]

    if title == None:
        title = y_column 
       
    if (x_column == 'index' and np.issubdtype(data.index, np.datetime64)) or np.issubdtype(data[x_column], np.datetime64):
        if x_column == 'index':
            plot_date(data.index, data[y_column], axes=axes, title=title, color=color_list)
        else:
            plot_date(data[x_column], data[y_column], axes=axes, title=title, color=color_list)
    elif axes == None:
        data.plot.scatter(x=x_column, y=y_column, color=color_list)
    else:
        axes.scatter(x=data[x_column], y=data[y_column], color=color_list)
        axes.set_title(title)
        axes.set_xlabel(x_column)
        axes.set_ylabel(y_column)


def plot_list_scatters(data, list_dict, subplot_cols=None, subplot_rows=None, starting_index=0, index_offset=0, fig=None,
                       axes=None):
    """ Plot several scatter plots contained in data depending on setting provided in list_dict """
    # Set a good relation rows/cols for the plot if not specified
    if subplot_rows is None or subplot_cols is None:
        subplot_rows, subplot_cols = get_subplot_rows_cols(len(list_dict), [3,4,5])
    
    # Resize for better visualization of subplots
    plt.rcParams['figure.figsize'] = [subplot_cols * 5, subplot_rows * 4]

    if fig is None or axes is None:
        fig, axes = plt.subplots(subplot_rows, subplot_cols, sharex=False, sharey=False)

    i = starting_index
    while i < len(list_dict):
        # Take into account the case of only one plot
        if subplot_rows * subplot_cols == 1:
            ax = axes
        elif subplot_rows == 1:
            ax = axes[(i + index_offset) % subplot_cols]
        else:
            ax = axes[(i + index_offset) // subplot_cols, (i + index_offset) % subplot_cols]

        info = list_dict[i]
        x_column = info['x_column']
        y_column = info['y_column']
        title = info.get('title')
        highlight_column = info.get('highlight_column')
        highlight_color = info.get('highlight_color', 'r')
        normal_color = info.get('normal_color', 'b')

        plot_scatter(data, x_column, y_column, title, ax, highlight_column, highlight_color, normal_color)

        i = i + 1

    fig.tight_layout()

    # Resize to original settings
    plt.rcParams['figure.figsize'] = [10, 6]