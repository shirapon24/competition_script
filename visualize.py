import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import pandas_profiling
import sweetviz as sv


def output_sweetviz(tra, val, output_filename="Report"):
    my_report = sv.compare([tra, "Train"], [val, "Test"])
    my_report.show_html(output_filename+".html")

def plot_venn_train_test(tra, val, col):
    """trainとtestのベン図をplotする
    """
    fig, ax = plt.subplots(figsize=(6,9))
    plt.title(col, fontsize=10)
    train_unique = tra[col].unique()
    test_unique = val[col].unique()
    common_num = len(set(train_unique) & set(test_unique))
    venn2(subsets=(len(train_unique)-common_num, len(test_unique)-common_num, common_num),set_labels=('Train', 'Test'))
    return fig, ax

def plot_importance(models, feat_train_df):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.
    
    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .mean()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:100]

    fig, ax = plt.subplots(figsize=(7, len(order) * .2))
    sns.boxenplot(data=feature_importance_df, y='column', x='feature_importance', order=order, ax=ax, palette='viridis')
    ax.grid()
    fig.tight_layout()
    return fig, ax