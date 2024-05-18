import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from typing import Literal

METRIC = Literal["acc", "accuracy", "auc", "f1"]


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.npdarray, *, metrics: list[METRIC]
) -> dict[str, float]:

    res: dict[METRIC, float] = {}
    for metric in metrics:
        match metric:
            case "accuracy" | "acc":
                res["accuracy"] = accuracy_score(y_true, y_pred)
            case "auc":
                res["auc"] = roc_auc_score(y_true, y_pred)
            case "f1":
                res["f1"] = f1_score(y_true, y_pred)
            case _:
                raise ValueError("Unknown metric")
    return res
