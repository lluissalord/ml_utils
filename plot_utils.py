import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from tqdm import tqdm_notebook
import seaborn as sns
from scipy.stats import norm

def get_subplot_rows_cols(n_cols = [3,4,5]):
    max_cols = max(n_cols)
    if len(features_names) // max_cols == 0:
        subplot_rows = 1
        subplot_cols = len(features_names)
    else:
        found = False
        better_res = max(n_cols)
        better_n = None
        for n in sorted(n_cols, reverse=True):
            res = len(features_names) % n
            if res == 0:
                subplot_rows = len(features_names) // n
                subplot_cols = n
                found = True
                break
            elif better_res > res:
                better_res = res
                better_n = n

        if not found:
            subplot_rows = len(features_names) // better_n + 1
            subplot_cols = better_n
    return subplot_rows, subplot_cols

def plot_histograms(df, column_names, df_types, max_value_counts, subplot_rows, subplot_cols,
                    starting_index=0, index_offset=1, fit=norm):
    # Resize for better visualization of subplots
    plt.rcParams['figure.figsize'] = [subplot_cols * 5, subplot_rows * 4]

    i = starting_index
    while i < len(column_names):
        column_name = column_names[i]
        plt.subplot(subplot_rows, subplot_cols, i + index_offset)

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


# Here we define a function to draw scatter plots where we can highlight points as the outliers
def plot_scatter(data, x_column, y_column, title=None, axes=None, highlight_column=None, highlight_color='r',
                 normal_color='b'):
    # print(data[highlight_column].astype(int))
    if highlight_column != None:
        data[highlight_column][data[highlight_column] == True] = highlight_color
        data[highlight_column][data[highlight_column] == False] = normal_color
        color_pd = data[highlight_column]
        # color_pd = data[highlight_column] * highlight_color + (1 - data[highlight_column]) * normal_color
        color_list = color_pd[color_pd.notnull()].values.tolist()
    else:
        color_list = len(data[y_column]) * [normal_color, ]

    if axes == None:
        data.plot.scatter(x=x_column, y=y_column, color=color_list)
    else:
        axes.scatter(x=data[x_column], y=data[y_column], color=color_list)
        if title == None:
            title = y_column
        axes.set_title(title)
        axes.set_xlabel(x_column)
        axes.set_ylabel(y_column)


def plot_list_scatters(data, list_dict, subplot_cols, subplot_rows, starting_index=0, index_offset=0, fig=None,
                       axes=None):
    # Resize for better visualization of subplots
    plt.rcParams['figure.figsize'] = [subplot_cols * 5, subplot_rows * 4]

    if fig is None or axes is None:
        fig, axes = plt.subplots(subplot_rows, subplot_cols, sharex=False, sharey=False)

    i = starting_index
    while i < len(list_dict):
        # Take into account the case of only one plot
        if subplot_rows * subplot_cols == 1:
            ax = axes
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