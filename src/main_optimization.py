import os
import warnings
import numpy as np
import optuna
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")

from optuna.storages import RetryFailedTrialCallback
from optuna.pruners import ThresholdPruner
from optuna.samplers import TPESampler
from modules.models import ModelSNNPC
from modules.other.utils import read_data

BASE_SEED = None
warnings.filterwarnings("ignore")
np.random.seed(BASE_SEED)
PATH = os.path.dirname(os.path.realpath(__file__))

def optimize_parameters(trial, dataset_name, train_dfs, test_dfs, hyperparameters, layers):
    betas = tuple(
        trial.suggest_float(f'beta{i+1}', 0.1, 1.0, log=True) for i in range(layers)
    ) if hyperparameters['beta'] is None else hyperparameters['beta']
    slope = trial.suggest_int('slope', 10, 50, step=1) if hyperparameters['slope'] is None else hyperparameters['slope']
    thresholds=tuple(
        trial.suggest_float(f'threshold{i+1}', 0.1, 1, log=True) for i in range(layers)
    ) if hyperparameters['threshold'] is None else hyperparameters['threshold']
    weight_minority_class = trial.suggest_float('weight', 0.95, 1, log=True) if hyperparameters['weight'] is None else hyperparameters['weight']
    class_weights = (1-weight_minority_class, weight_minority_class) 
    adam_betas = tuple( 
        trial.suggest_float(f'adam_beta{i+1}', 0.97, 0.99, step=0.001) for i in range(2)
    ) if hyperparameters['adam_beta'] is None else hyperparameters['adam_beta']
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True) if hyperparameters['learning_rate'] is None else hyperparameters['learning_rate']
    train_df = train_dfs[dataset_name].iloc[:, :32]
    test_df = test_dfs[dataset_name].iloc[:, :32]
    x_train = train_df.drop(columns=["fraud_bool"])
    y_train = train_df["fraud_bool"]
    x_test = test_df.drop(columns=["fraud_bool"])
    y_test = test_df["fraud_bool"]
    num_classes = len(np.unique(y_train))
    num_features = len(x_train.columns)
    model = ModelSNNPC(
        num_features=num_features,
        num_classes=num_classes,
        population=hyperparameters['population'],
        class_weights=class_weights,
        betas=betas,
        slope=slope,
        thresholds=thresholds,
        batch_size=hyperparameters['batch'],
        num_epochs=hyperparameters['epoch'],
        num_steps=hyperparameters['step'],
        adam_betas=adam_betas,
        learning_rate=learning_rate,
        verbose=0
    )
    fit_time = time.time()
    model.fit(x_train, y_train)
    trial.set_user_attr("@time train", time.time()-fit_time)
    inference_time = time.time()
    predictions, targets = model.predict(x_test, y_test)
    trial.set_user_attr("@time inference", time.time()-inference_time)
    metrics = model.evaluate(targets, predictions)
    metrics_business = model.evaluate_business_constraint(targets, predictions)
    metrics.update(metrics_business)
    fairness_age = model.evaluate_fairness(x_test, targets, predictions, "customer_age", 50)
    metrics.update({k+"_age": v for k, v in fairness_age.items()})
    fairness_income = model.evaluate_fairness(x_test, targets, predictions, "income", 0.5)
    metrics.update({k+"_income": v for k, v in fairness_income.items()})
    fairness_employement = model.evaluate_fairness(x_test, targets, predictions, "employment_status", 3)
    metrics.update({k+"_employment": v for k, v in fairness_employement.items()})
    trial.set_user_attr("@global accuracy", metrics["accuracy"])
    trial.set_user_attr("@global precision", metrics["precision"])
    trial.set_user_attr("@global recall", metrics["recall"])
    trial.set_user_attr("@global fpr", metrics["fpr"])
    trial.set_user_attr("@global f1_score", metrics["f1_score"])
    trial.set_user_attr("@global auc", metrics["auc"])
    try:
        trial.set_user_attr("@5FPR fpr", metrics["fpr@5FPR"])
        trial.set_user_attr("@5FPR recall", metrics["recall@5FPR"])
        trial.set_user_attr("@5FPR accuracy", metrics["accuracy@5FPR"])
        trial.set_user_attr("@5FPR precision", metrics["precision@5FPR"])
        trial.set_user_attr("@5FPR threshold", metrics["threshold"])
        trial.set_user_attr("@5FPR fpr_ratio_age", metrics["fpr_ratio_age"])
        trial.set_user_attr("@5FPR fpr_ratio_income", metrics["fpr_ratio_income"])
        trial.set_user_attr("@5FPR fpr_ratio_employment", metrics["fpr_ratio_employment"])
    except Exception:
        pass
    objectives = [metrics[y] for (_,y) in OBJECTIVE]
    return objectives

def main(datasets_list, study_name, trials_optuna, sampler, objective, hyperparameters, layers=3):
    base_path = f"{PATH}/../data/"
    _, datasets, train_dfs, test_dfs = read_data(base_path, datasets_list, seed=BASE_SEED)
    for dataset_name in datasets.keys(): 
        storage = optuna.storages.RDBStorage(
            url="sqlite:///ciarp.db",
            heartbeat_interval=60,
            grace_period=120,
            failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        ) 
        study = optuna.create_study(
            directions=[x for (x,_) in objective],
            storage=storage,
            load_if_exists=True,
            study_name=f"{study_name}",
            sampler=sampler,
            pruner=ThresholdPruner(lower=0.01, upper=0.99)
        )
        study.optimize(lambda trial, dataset_name=dataset_name: optimize_parameters(trial, dataset_name, train_dfs, test_dfs, hyperparameters, layers), n_trials=trials_optuna)
        try:
            print(study.best_params)
            print(study.best_value)
            print(study.best_trial)
        except Exception:
            pass



if __name__ == "__main__":
    HYPERPARAMETERS = {
        "population": 2,
        "batch": 1024,
        "epoch": 5,
        "step": 20,
        "beta": None,
        "slope": None,
        "threshold": None,
        "weight": None,
        "adam_beta": None,
        "learning_rate": None,
    }
    LAYERS = 4
    DATASETS = ["Base"]
    STUDY_NAME = f"CIARP2024-P{HYPERPARAMETERS['population']}-S{HYPERPARAMETERS['step']}"
    TRIALS_OPTUNA = 1050
    SAMPLER = TPESampler()
    OBJECTIVE = [("minimize","fpr"), ("maximize","recall")]
main(DATASETS, STUDY_NAME, TRIALS_OPTUNA, SAMPLER, OBJECTIVE, HYPERPARAMETERS, LAYERS)
