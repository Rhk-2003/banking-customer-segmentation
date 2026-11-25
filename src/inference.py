"""Inference surface for the segmentation app, so the dashboard and the training
script can never drift apart in how a customer is encoded."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import log_transform

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def load_segmentation():
    pipe = joblib.load(MODELS / "segmentation_pipeline.pkl")
    meta = json.loads((MODELS / "segmentation_meta.json").read_text())
    profile = pd.read_csv(ROOT / "reports" / "segment_profile.csv")
    return pipe, meta, profile


def predict_segment(pipe, meta, values: dict) -> dict:
    """values: raw (un-logged) feature values keyed by meta['cluster_cols']."""
    cols = meta["cluster_cols"]
    X = pd.DataFrame([[values[c] for c in cols]], columns=cols)
    Xl = log_transform(X)
    seg = int(pipe.predict(Xl)[0])

    # distance to every centroid, so the UI can show how clear-cut the call is
    Z = pipe.named_steps["scaler"].transform(Xl)
    centers = pipe.named_steps["kmeans"].cluster_centers_
    d = np.linalg.norm(centers - Z, axis=1)
    inv = 1.0 / (d + 1e-9)
    conf = inv / inv.sum()
    names = meta["segment_names"]
    return {
        "segment": seg,
        "label": names[str(seg)]["label"],
        "action": names[str(seg)]["action"],
        "distances": d,
        "confidence": conf,
        "margin": float(np.sort(d)[1] - np.sort(d)[0]),
    }
