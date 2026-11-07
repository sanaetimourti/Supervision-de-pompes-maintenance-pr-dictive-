"""Métriques officielles UC5."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def score_nasa(y_pred, y_true):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def metriques_rul(rul_pred, rul_vraie, par_pompe=True):
    p = np.clip(np.asarray(rul_pred, float), 0, 125)
    t = np.minimum(np.asarray(rul_vraie, float), 125)
    s = score_nasa(p, t)
    sig = p < 30
    return {
        "RMSE (cycles)": float(np.sqrt(np.mean((p - t) ** 2))),
        "Score NASA": s if par_pompe else 30 * s / len(t),
        "Anticipation": float(sig[t <= 30].mean()) if np.any(t <= 30) else np.nan,
        "Fausses alertes": float(sig[t > 60].mean()) if np.any(t > 60) else np.nan,
    }


def metriques_mode(y_true, y_pred, labels=(0, 1, 2, 3)):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, labels=labels,
                             average="macro", zero_division=0),
    }