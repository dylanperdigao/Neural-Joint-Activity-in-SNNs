import seaborn as sns   
import numpy as np
import os

from copy import deepcopy
from matplotlib import pyplot as plt   



def performance_fairness(df, metrics, filename="performance_fairness.png", path=".", export_eps=False, export_png=False, **kwargs):
    """
    Plot performance vs fairness
    ----------
    Parameters
    ----------
        df : pandas.DataFrame
            dataframe with the data
        metrics : list
            list with the metrics
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
            title of the plot (default is "Performance vs Fairness")
        x_name : str, optional
            name of the x axis (default is "xlabel")
        y_name : str, optional
            name of the y axis (default is "ylabel")
        hue_name : str, optional
            name of the hue (default is "zlabel")
        text_scale : float, optional
            scale of the text (default is 1)
        alpha : float, optional
            transparency of the points (default is 1)
        size : int, optional
            size of the points (default is 50)
        loc : str, optional
            location of the legend (default is "upper right")
    """
    title = kwargs.get("title", "Performance vs Fairness")
    x_name = kwargs.get("x_name", "xlabel")
    y_name = kwargs.get("y_name", "ylabel")
    z_name= kwargs.get("hue_name", "zlabel")
    text_scale = kwargs.get("text_scale", 1)
    alpha = kwargs.get("alpha", 1)
    size = kwargs.get("size", 50)
    loc = kwargs.get("loc", "upper right")
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    plt.figure(figsize=(5, 5))
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.5)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.5)
    df = df.dropna()
    sns.jointplot(data=df, x=metrics[0], y=metrics[1], hue=metrics[2], alpha=alpha, joint_kws={'s': size})
    plt.title(title, y=1.2)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.legend(title=z_name, loc=loc)
    if export_png:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=300, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
        