"""Fonctions de prétraitement partagées entre les 4 jalons."""
import numpy as np
import pandas as pd

RUL_MAX = 125
CAPTEURS = [f"c{i}" for i in range(8)]


def charger_flotte(path):
    """Charge le NPZ et renvoie un dict de DataFrames bruts."""
    d = np.load(path, allow_pickle=False)

    def build(split):
        df = pd.DataFrame({
            "pompe": d[f"pompe_{split}"].astype(int),
            "cycle": d[f"cycle_{split}"].astype(int),
            "charge": d[f"charge_{split}"].astype("float32"),
        })
        cap = d[f"capteurs_{split}"].astype("float32")
        for i, c in enumerate(CAPTEURS):
            df[c] = cap[:, i]
        if f"rul_{split}" in d.files:
            df["rul"] = d[f"rul_{split}"].astype(int)
            df["mode"] = d[f"mode_{split}"].astype(int)
        return df.sort_values(["pompe", "cycle"]).reset_index(drop=True)

    TA = {s: build(s) for s in ["train", "val", "test"]}
    meta = {
        "spectro": {s: d[f"spectro_{s}"].astype("float32") / 255.0
                    for s in ["train", "val", "test"]},
        "pompe_id_test": d["pompe_id_test"].astype(int),
        "rul_fin_test": d["rul_fin_test"].astype(int),
        "mode_pompe_test": d["mode_pompe_test"].astype(int),
    }
    return TA, meta


def fit_normalisation(TA):
    """Statistiques par régime sur les cycles sains du train (RUL > 125)."""
    sains = TA["train"][TA["train"]["rul"] > RUL_MAX]
    mu = sains.groupby("charge")[CAPTEURS].mean()
    sd = sains.groupby("charge")[CAPTEURS].std().replace(0, 1).fillna(1)
    return mu, sd


def normaliser(df, mu, sd):
    out = df.copy()
    for charge, idx in out.groupby("charge").groups.items():
        if charge in mu.index:
            out.loc[idx, CAPTEURS] = (
                out.loc[idx, CAPTEURS].values - mu.loc[charge].values
            ) / sd.loc[charge].values
    return out


def fenetres_aplaties(df, k=1, cap=RUL_MAX):
    """Fenêtres de k cycles aplaties pour MLP / CNN."""
    X, y, fin, pompe, cycle = [], [], [], [], []
    for pid, g in df.sort_values(["pompe", "cycle"]).groupby("pompe"):
        v = g[CAPTEURS + ["charge"]].values.astype("float32")
        if k > 1:
            v = np.vstack([np.repeat(v[:1], k - 1, axis=0), v])
        X.append(np.stack([v[i:i + k].ravel() for i in range(len(g))]))
        fin.append(np.arange(len(g)) == len(g) - 1)
        pompe.append(np.repeat(pid, len(g)))
        cycle.append(g["cycle"].values)
        if "rul" in g:
            y.append(np.minimum(g["rul"].values, cap) / cap)
    return (np.concatenate(X).astype("float32"),
            np.concatenate(y).astype("float32") if y else None,
            np.concatenate(fin), np.concatenate(pompe), np.concatenate(cycle))


def sequences_rnn(df, W=30, zeros_masking=True):
    """Fenêtres séquentielles (N, W, 9) pour LSTM/GRU."""
    X, y, fin, pompe = [], [], [], []
    for pid, g in df.sort_values(["pompe", "cycle"]).groupby("pompe"):
        v = g[CAPTEURS + ["charge"]].values.astype("float32")
        n = len(v)
        if zeros_masking:
            pad = np.zeros((W - 1, v.shape[1]), dtype="float32")
        else:
            pad = np.repeat(v[:1], W - 1, axis=0)
        vp = np.vstack([pad, v])
        X.append(np.stack([vp[i:i + W] for i in range(n)]))
        fin.append(np.arange(n) == n - 1)
        pompe.append(np.repeat(pid, n))
        if "rul" in g:
            y.append(np.minimum(g["rul"].values, RUL_MAX) / RUL_MAX)
    return (np.concatenate(X).astype("float32"),
            np.concatenate(y).astype("float32") if y else None,
            np.concatenate(fin), np.concatenate(pompe))