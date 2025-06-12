from __future__ import annotations

import pathlib, sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 13
PCA_VARIANCE_THRESHOLD = 0.95
MISSING_STRATEGY = "mean"
SAVE_PLOTS = True

PUBLIC_DATASET = pathlib.Path("public_data.csv")
PRIVATE_DATASET = pathlib.Path("private_data.csv")

def load_dataset(path: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    id_col = next((c for c in df.columns if str(c).lower() == "id"), None)
    ids = df[id_col] if id_col else pd.Series(np.arange(1, len(df) + 1), name="id")
    feat_df = df.drop(columns=[id_col]) if id_col else df
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    feat_df = feat_df.fillna(feat_df.mean() if MISSING_STRATEGY == "mean" else 0.0)
    return feat_df.astype(np.float64), ids

def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[PCA]]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df).astype(np.float64)
    if df.shape[1] > 4:
        pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full", random_state=RANDOM_STATE)
        return pca.fit_transform(X_scaled), pca
    return X_scaled, None

def _try_gmm(X: np.ndarray, k: int, cov: str, reg: float) -> Optional[np.ndarray]:
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        reg_covar=1e-3,
        init_params="random",
        n_init=10,
        random_state=RANDOM_STATE,
        max_iter=500,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gmm.fit(X)
        return gmm.predict(X)
    except Exception:
        return None

def cluster_data(X: np.ndarray, n_dims: int) -> np.ndarray:
    k = 4 * n_dims - 1
    for reg in (1e-6, 1e-4, 1e-2):
        labels = _try_gmm(X, k, "full", reg)
        if labels is not None:
            return labels
    labels = _try_gmm(X, k, "diag", 1e-3)
    if labels is not None:
        return labels
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto", max_iter=300)
    return km.fit_predict(X)

def _project_to_2d(X: np.ndarray) -> Tuple[np.ndarray, str]:
    d = X.shape[1]
    if d == 1:
        return np.column_stack([np.arange(len(X)), X[:, 0]]), "index-feature"
    if d == 2:
        return X, "original-2D"
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca2.fit_transform(X), "PCA-2D"

def save_cluster_plot(X: np.ndarray, labels: np.ndarray, filename: str) -> None:
    if not SAVE_PLOTS:
        return
    X2, _ = _project_to_2d(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = get_cmap("tab20")
    uniq = np.unique(labels)
    for idx, lab in enumerate(uniq):
        pts = X2[labels == lab]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.6, color=cmap(idx % cmap.N))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def process_one(path: pathlib.Path, out_csv: str) -> None:
    feat_df, ids = load_dataset(path)
    n_dims = feat_df.shape[1]
    X_pre, _ = preprocess(feat_df)
    labels = cluster_data(X_pre, n_dims)
    pd.DataFrame({"id": ids.values, "label": labels}).to_csv(out_csv, index=False)
    save_cluster_plot(X_pre, labels, pathlib.Path(out_csv).with_suffix(".png"))

def main(argv: List[str] | None = None) -> None:
    process_one(PUBLIC_DATASET, "public_submission.csv")
    process_one(PRIVATE_DATASET, "private_submission.csv")

if __name__ == "__main__":
    main()
