import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def tradeoff_performance_fairness(df, performance_metric="performance", fairness_metric="fairness", filename="tradeoff_performance_fairness.png", path=".", export_eps=False, export_png=False, **kwargs):
    """
    Plot tradeoff between performance and fairness
    ----------
    Parameters
    ----------
        df : pandas.DataFrame
            dataframe with the data
        performance_metric : str, optional
            name of the performance metric (default is "performance")
        fairness_metric : str, optional
            name of the fairness metric (default is "fairness")
        filename : str, optional
            name of the file
        path : str, optional
            path to save the plot
        export_eps: bool, optional
            boolean to export eps file
        export_png: bool, optional
            boolean to export png file
        **kwargs:
            see below

    Keyword Arguments
    ----------
        title : str, optional
            title of the plot (default is None)
        text_scale : float, optional
            scale of the text (default is 1)
        loc : str, optional
            location of the legend (default is "upper right")
        dpi : int, optional
            resolution of the image (default is 1000)
        x_label : str, optional
            name of the x axis (default is None)
        y_label : str, optional
            name of the y axis (default is None)
    """
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    text_scale = kwargs.get("text_scale", 1)
    loc = kwargs.get("loc", "upper right")
    dpi = kwargs.get("dpi", 1000)
    sns.set_style("darkgrid")
    sns.set_context("paper") 
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.5)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.5)
    plt.figure(figsize=(10, 6))
    tradeoff_list = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        for alpha in range(0, 101, 1):
            alpha = alpha / 100
            tradeoff = alpha * model_df[performance_metric].mean() + (1 - alpha) * model_df[fairness_metric].mean()
            std_dev = ((alpha * model_df[performance_metric]).std()**2 + ((1 - alpha) * model_df[fairness_metric]).std()**2)**0.5
            tradeoff_list.append({"model": model, "alpha": alpha, "tradeoff": tradeoff, "std_dev": std_dev})
    tradeoff_df = pd.DataFrame(tradeoff_list)
    sns.lineplot(data=tradeoff_df, x="alpha", y="tradeoff", hue="model")
    for name, group in tradeoff_df.groupby("model"):
        plt.fill_between(group['alpha'], group['tradeoff'] - group['std_dev'], group['tradeoff'] + group['std_dev'], alpha=0.1)
    if loc == "top":
        plt.legend(loc="lower left", bbox_to_anchor=(-0.05, 1), ncol=4, title=None, frameon=False)
    else:
        plt.legend(title="Model", loc=loc)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(kwargs.get("x_label")) if kwargs.get("x_label") else plt.xlabel("\u03B1")
    plt.ylabel(kwargs.get("y_label")) if kwargs.get("y_label") else plt.ylabel(f"\u03B1 $\cdot$ {performance_metric} + (1-\u03B1) $\cdot$ {fairness_metric}")
    plt.title(kwargs.get("title"), y=1.2) if kwargs.get("title") else None
    if export_png:
        plt.savefig(f'{path}/{filename}.{extension}', dpi={dpi}, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi={dpi}, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")