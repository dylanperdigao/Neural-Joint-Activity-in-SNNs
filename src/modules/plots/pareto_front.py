import matplotlib.pyplot as plt
import seaborn as sns
import os

def pareto_front(study, filename="pareto_front.png", path=".", export_eps=False, export_png=False, **kwargs):
    """
    Plot the pareto front of a study
    ----------
    Parameters
    ----------
        study : optuna.study.Study
            optuna study object
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
            title of the plot (default is "Pareto Front")
        text_scale : float, optional
            scale of the text (default is 1)
        loc : str, optional
            location of the legend (default is "upper right")
        size : int, optional
            size of the points (default is 5)
    """
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    title = kwargs.get("title", "Pareto Front")
    text_scale = kwargs.get("text_scale", 1)
    loc = kwargs.get("loc", "upper right")
    size = kwargs.get("size", 5)
    df = study.trials_dataframe()
    best_trials = study.best_trials
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.5)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.5)
    df_best = df[df['number'].isin([trial.number for trial in best_trials])]
    df_not_best = df[~df['number'].isin([trial.number for trial in best_trials])]
    plt.subplots(figsize=(10, 5))
    plt.title(title)
    plt.scatter(df_not_best['user_attrs_@global fpr'], df_not_best['user_attrs_@global recall'], c=df_not_best['number'], cmap='Blues', s=size)
    plt.scatter(df_best['user_attrs_@global fpr'], df_best['user_attrs_@global recall'], c=df_best['number'], cmap='Reds', s=size)
    plt.plot([0.05, 0.05], [0,1], 'k--')
    plt.xlim(0, 0.25)
    plt.ylim(0, 1)
    plt.xlabel('FPR')
    plt.ylabel('Recall')
    leg = plt.legend(['Trials', 'Best trials', "5% FPR"], loc=loc)
    leg.legendHandles[0].set_color('darkblue')
    leg.legendHandles[1].set_color('darkred')
    leg.legendHandles[2].set_color('black')
    if export_png:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=300, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")