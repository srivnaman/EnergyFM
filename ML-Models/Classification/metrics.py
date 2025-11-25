
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import logging


def compute_metrics(y_true, y_pred, clf_name):
    """
    Compute classification metrics for a given set of predictions.

    Parameters:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted labels.
        clf_name (str): Name of the classifier used.

    Returns:
        dict: Dictionary containing classification metrics.
    """
    try:
        metrics = {                    
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "PrecisionMacro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "RecallMacro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1Score": f1_score(y_true, y_pred, average="binary", zero_division=0),
            "F1ScoreMacro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "F1ScoreWeighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, y_pred),
            "ROC_AUC_Macro": roc_auc_score(y_true, y_pred, average="macro"),
            "ROC_AUC_Weighted": roc_auc_score(y_true, y_pred, average="weighted"),
        }
        
        metrics_rounded = {key: round(value, 5) for key, value in metrics.items()}
        return metrics_rounded        
        
    except Exception as e:
        logging.error(f"[{clf_name}] Metric computation failed: {e}")
        return {"Model": clf_name}
