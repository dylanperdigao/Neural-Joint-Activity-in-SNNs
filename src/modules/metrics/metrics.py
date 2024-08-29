import pandas as pd
import numpy as np
from aequitas.group import Group    
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def evaluate(y_test, predicted):
    """Evaluate the model using the confusion matrix and some metrics.
    ------------------------------------------------------ 
    Args:
        targets (list): list of true values
        predicted (list): list of predicted values
    ------------------------------------------------------ 
    Returns:
        cm (np.array): confusion matrix
        tn (int): true negative
        fp (int): false positive
        fn (int): false negative
        tp (int): true positive
        accuracy (float): accuracy of the model
        precision (float): precision of the model
        recall (float): recall of the model
        fpr (float): false positive rate of the model
        tnr (float): true negative rate of the model
        f1_score (float): f1 score of the model
        auc (float): area under the curve of the model
    """
    y_pred = []
    y_true = []
    y_pred.extend(predicted)
    y_true.extend(y_test)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = 0 if (tp + tn + fp + fn)==0 else (tp + tn) / (tp + tn + fp + fn)
    precision = 0 if (tp + fp)==0 else tp / (tp + fp) 
    recall = 0 if (tp + fn)==0 else tp / (tp + fn)
    fpr = 0 if (fp + tn)==0 else fp / (fp + tn)
    tnr = 0 if (tn + fp)==0 else tn / (tn + fp)
    f1 = 0 if (precision + recall)==0 else 2 * (precision * recall) / (precision + recall) 
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0
    return {
        "cm": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "recall": recall,
        "tnr": tnr,
        "accuracy": accuracy,
        "precision": precision,
        "f1_score": f1,
        "auc": auc
    }

def evaluate_business_constraint(y_test, predictions):
    """Evaluate the model using the Aequitas library.
    ------------------------------------------------------
    Args:
        y_test (pd.Series): series with the test labels
        predictions (np.array): array with the predictions
    ------------------------------------------------------
    Returns:
        threshold (float): threshold for the model
        fpr@5FPR (float): false positive rate of the model
        recall@5FPR (float): recall of the model
        tnr@5FPR (float): true negative rate of the model
        accuracy@5FPR (float): accuracy of the model
        precision@5FPR (float): precision of the model
        f1_score@5FPR (float): f1 score of the model
    """
    fprs, _, thresholds = roc_curve(y_test, predictions)    
    threshold = np.min(thresholds[fprs==max(fprs[fprs < 0.05])])
    preds_binary = (predictions >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_binary).ravel()
    cm_recall = 0 if (tp + fn)==0 else tp / (tp + fn)
    cm_tnr = 0 if (tn + fp)==0 else tn / (tn + fp)
    cm_accuracy = 0 if (tp + tn + fp + fn)==0 else (tp + tn) / (tp + tn + fp + fn)
    cm_precision = 0 if (tp + fp)==0 else tp / (tp + fp) 
    cm_fpr = 0 if (fp + tn)==0 else fp / (fp + tn)
    cm_f1 = 0 if (cm_precision + cm_recall)==0 else 2 * (cm_precision * cm_recall) / (cm_precision + cm_recall) 
    return {
        "threshold": threshold,
        "fpr@5FPR": cm_fpr,
        "recall@5FPR": cm_recall,
        "tnr@5FPR": cm_tnr,
        "accuracy@5FPR": cm_accuracy,
        "precision@5FPR": cm_precision,
        "f1_score@5FPR": cm_f1,
    }
    
def evaluate_fairness(x_test, y_test, predictions, sensitive_attribute, attribute_threshold, all_metrics=False, variable_name=None):
    """
    Evaluate the fairness of the model using the Aequitas library.
    ------------------------------------------------------
    Args:
        x_test (pd.DataFrame): dataframe with the test features
        y_test (pd.Series): series with the test labels
        predictions (np.array): array with the predictions
        sensitive_attribute (str): name of the sensitive attribute
        attribute_threshold (int): threshold for the sensitive attribute (g1 is below the threshold and g2 is above)
        all_metrics (bool): flag to return all metrics
        variable_name (str): name of the variable
    ------------------------------------------------------
    Returns:
        fpr_ratio (float): false positive rate ratio of the model
        fnr_ratio (float): false negative rate ratio of the model
        recall_g2 (float): recall of the group 2
        recall_g1 (float): recall of the group 1
        fpr_g2 (float): false positive rate of the group 2
        fpr_g1 (float): false positive rate of the group 1
        fnr_g2 (float): false negative rate of the group 2
        fnr_g1 (float): false negative rate of the group 1
    """

    fprs, _, thresholds = roc_curve(y_test, predictions)    
    threshold = np.min(thresholds[fprs==max(fprs[fprs < 0.05])])
    preds_binary = (predictions >= threshold).astype(int)
    aequitas_df = pd.DataFrame(
        {
            "attribute": (x_test[sensitive_attribute]>=attribute_threshold).map({True: "g2", False: "g1"}),
            "preds": preds_binary,
            "y": y_test.values if isinstance(y_test, pd.Series) else y_test
        }
    )
    g = Group()
    aequitas_df["score"] = aequitas_df["preds"]
    aequitas_df["label_value"] = aequitas_df["y"]
    aequitas_results = g.get_crosstabs(aequitas_df, attr_cols=["attribute"])[0]
    recall_g2 = aequitas_results[aequitas_results["attribute_value"] == "g2"][["tpr"]].values[0][0]
    recall_g1 = aequitas_results[aequitas_results["attribute_value"] == "g1"][["tpr"]].values[0][0]
    fpr_g2 = aequitas_results[aequitas_results["attribute_value"] == "g2"][["fpr"]].values[0][0]
    fpr_g1 = aequitas_results[aequitas_results["attribute_value"] == "g1"][["fpr"]].values[0][0]
    fnr_g2 = 1 - recall_g2
    fnr_g1 = 1 - recall_g1
    if fpr_g1 >= fpr_g2:
        fpr_ratio = fpr_g1 and fpr_g2/fpr_g1 or 0
    else:
        fpr_ratio = fpr_g2 and fpr_g1/fpr_g2 or 0
    if fnr_g1 > fnr_g2:
        fnr_ratio = fnr_g2 and fnr_g2/fnr_g1 or 0
    else:
        fnr_ratio = fnr_g2 and fnr_g1/fnr_g2 or 0
    if variable_name is not None:
        if all_metrics:
            return {f"fpr_ratio_{variable_name}": fpr_ratio, f"fnr_ratio_{variable_name}": fnr_ratio, f"recall_{variable_name}_g2": recall_g2, f"recall_{variable_name}_g1": recall_g1, f"fpr_{variable_name}_g2": fpr_g2, f"fpr_{variable_name}_g1": fpr_g1, f"fnr_{variable_name}_g2": fnr_g2, f"fnr_{variable_name}_g1": fnr_g1}
        return {f"fpr_ratio_{variable_name}": fpr_ratio, f"fnr_ratio_{variable_name}": fnr_ratio}
    if all_metrics:
        return {"fpr_ratio": fpr_ratio, "fnr_ratio": fnr_ratio, "recall_g2": recall_g2, "recall_g1": recall_g1, "fpr_g2": fpr_g2, "fpr_g1": fpr_g1, "fnr_g2": fnr_g2, "fnr_g1": fnr_g1}
    return {"fpr_ratio": fpr_ratio, "fnr_ratio": fnr_ratio}
