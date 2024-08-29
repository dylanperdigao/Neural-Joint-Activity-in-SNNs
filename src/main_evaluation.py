import numpy as np
import os
import random   

from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from modules.other.utils import read_data, RandomTrial
from modules.other import hyperparameters_ciarp2024
from modules.models import  ModelSNNPC

DATASET_LIST = ["Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"]
NUM_TRIALS = 100
BEGIN_TRIAL = 0
BASE_SEED = 42

PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_NAME_GLOBAL = ["accuracy", "precision", "recall", "fpr", "f1_score","auc"]
METRICS_NAME_5FPR = ["accuracy@5FPR","precision@5FPR", "recall@5FPR", "fpr@5FPR", "f1_score@5FPR"]
METRICS_FAIRNESS = ["fpr_ratio_age", "fpr_ratio_income", "fpr_ratio_employment"]

HYPERPARAMETERS = hyperparameters_ciarp2024.P2_S20
#HYPERPARAMETERS = hyperparameters_ciarp2024.P20_S20
#HYPERPARAMETERS = hyperparameters_ciarp2024.P200_S20
#HYPERPARAMETERS = hyperparameters_ciarp2024.P2_S50
#HYPERPARAMETERS = hyperparameters_ciarp2024.P20_S50
#HYPERPARAMETERS = hyperparameters_ciarp2024.P200_S50

EXPERIMENT_NAME = f"CIARP2024-P{HYPERPARAMETERS['population']}-S{HYPERPARAMETERS['step']}-{NUM_TRIALS}trials-begin{BEGIN_TRIAL}"
FIXED_DATE = None 

def dataset_loop(train_dfs, test_dfs, dataset_name, trial_number, seed, path, runs):
    x_train = train_dfs[dataset_name].drop(columns=["fraud_bool"])
    y_train = train_dfs[dataset_name]["fraud_bool"]
    x_test = test_dfs[dataset_name].drop(columns=["fraud_bool"])
    y_test = test_dfs[dataset_name]["fraud_bool"]
    num_classes = len(np.unique(y_train))
    num_features = len(x_train.columns)
    class_weights = (1-HYPERPARAMETERS['weight'], HYPERPARAMETERS['weight'])
    model = ModelSNNPC(
        num_features=num_features,
        num_classes=num_classes,
        class_weights=class_weights,
        betas=HYPERPARAMETERS['beta'],
        slope=HYPERPARAMETERS['slope'],
        thresholds=HYPERPARAMETERS['threshold'],
        population=HYPERPARAMETERS['population'],
        batch_size=HYPERPARAMETERS['batch'],
        num_epochs=HYPERPARAMETERS['epoch'],
        num_steps=HYPERPARAMETERS['step'],
        adam_betas=HYPERPARAMETERS['adam_beta'],
        learning_rate=HYPERPARAMETERS['learning_rate'],
        verbose=0
    )
    model.fit(x_train, y_train)
    predictions, targets = model.predict(x_test, y_test)
    metrics = model.evaluate(targets, predictions)
    metrics_aequitas = model.evaluate_business_constraint(targets, predictions)
    metrics.update(metrics_aequitas)
    fairness_age = model.evaluate_fairness(x_test, targets, predictions, "customer_age", 50)
    metrics.update({k+"_age": v for k, v in fairness_age.items()})
    fairness_income = model.evaluate_fairness(x_test, targets, predictions, "income", 0.5)
    metrics.update({k+"_income": v for k, v in fairness_income.items()})
    fairness_employement = model.evaluate_fairness(x_test, targets, predictions, "employment_status", 3)
    metrics.update({k+"_employment": v for k, v in fairness_employement.items()})
    results = {}
    results["dataset"] = dataset_name
    results["trial"] = trial_number
    results["seed"] = seed
    for metric in METRICS_NAME_GLOBAL:
        results[metric] = metrics[metric]
    for metric in METRICS_NAME_5FPR:
        results[metric] = metrics_aequitas[metric]
    for metric in METRICS_FAIRNESS:
        results[metric] = metrics[metric]
    csv_row = ','.join([str(x) for x in results.values()])
    with open(path, "a") as f:
        f.write(f"{csv_row}\n")
    prev_runs = runs.get(dataset_name, [])
    prev_runs.append(results)
    runs[dataset_name] = prev_runs
    return runs

def simulation(datasets, train_dfs, test_dfs, path="./results.csv"):
    np.random.seed(BASE_SEED)
    seeds = np.random.choice(list(range(1_000_000)), size=NUM_TRIALS, replace=False)
    runs = {}
    for trial in range(NUM_TRIALS):
        seed = seeds[trial]
        trial_number = trial
        trial = RandomTrial(seed=seed)
        if trial_number < BEGIN_TRIAL:
            print(f"Skipping trial {trial_number} – seed {seed}")
            continue
        for dataset_name in datasets.keys():
            print(f"Running trial {trial_number} with seed {seed} on dataset {dataset_name}")
            dataset_loop(train_dfs, test_dfs, dataset_name, trial_number, seed, path, runs)   

def main():
    base_path = f"{PATH}/../data/"
    _, datasets, train_dfs, test_dfs = read_data(base_path, DATASET_LIST)
    if not FIXED_DATE:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        date = FIXED_DATE
    experiment_dir = f"{PATH}/results/{date}-{EXPERIMENT_NAME}"
    results_path = f"{experiment_dir}/results.csv"
    os.makedirs(experiment_dir, exist_ok=True)
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("dataset,trial,seed,accuracy,precision,recall,fpr,f1_score,auc,accuracy@5FPR,precision@5FPR,recall@5FPR,fpr@5FPR,f1_score@5FPR,fpr_ratio_age,fpr_ratio_income,fpr_ratio_employment\n")
    simulation(datasets, train_dfs, test_dfs, path=results_path)
    
if __name__ == "__main__":
    main()
