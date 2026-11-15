"""Loaders and validators for the Elliptic and Elliptic++ datasets.

Every count this project reports is computed here from the checksummed files on disk. Figures
quoted in the literature are treated as claims to verify, not as inputs: several widely-repeated
numbers for these datasets disagree with each other, so the manuscript uses only what the files
themselves say.

File layout
-----------
``elliptic_txs_features.csv`` has **no header row**. Field 0 is the transaction id, field 1 is
the time step, and the remaining fields are the anonymised features. Loading it with a header
silently discards the first transaction, which is a real and easy mistake, so the loader passes
``header=None`` explicitly and asserts the field count.

``elliptic_txs_classes.csv`` labels transactions ``1`` (illicit), ``2`` (licit) or ``unknown``.
The project maps these to ``1``, ``0`` and ``-1`` respectively; ``-1`` rows are excluded from
supervised evaluation but retained in the graph, because removing them would sever the paths
that carry the structure being measured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ILLICIT, LICIT, UNKNOWN = 1, 0, -1
CLASS_MAP = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}

#: Column names assigned to the headerless features file.
TXID = "txId"
TIMESTEP = "time_step"


@dataclass
class EllipticData:
    """The Elliptic transaction dataset as loaded from disk.

    Attributes:
        features: one row per transaction, indexed by ``txId``, with ``time_step`` as the first
            column followed by ``feat_001 ...``.
        edges: two columns, ``txId1`` and ``txId2``, directed transaction-to-transaction flow.
        labels: Series indexed by ``txId`` with values in ``{1, 0, -1}``.
    """

    features: pd.DataFrame
    edges: pd.DataFrame
    labels: pd.Series

    @property
    def n_nodes(self) -> int:
        return len(self.features)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.features.columns if c.startswith("feat_")]


def load_elliptic(raw_dir: Path, *, float_dtype: str = "float32") -> EllipticData:
    """Load the Elliptic dataset from ``raw_dir``.

    Args:
        raw_dir: directory holding the three ``elliptic_txs_*.csv`` files.
        float_dtype: dtype for the feature matrix. ``float32`` halves memory at no cost to the
            analysis, since the features are already standardised to roughly unit scale.

    Raises:
        FileNotFoundError: if any of the three files is missing.
        ValueError: if the feature file's shape is not what the format requires.
    """
    features_path = raw_dir / "elliptic_txs_features.csv"
    edges_path = raw_dir / "elliptic_txs_edgelist.csv"
    classes_path = raw_dir / "elliptic_txs_classes.csv"
    for path in (features_path, edges_path, classes_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run data/download_scripts/get_elliptic.py")

    logger.info("reading %s", features_path.name)
    raw = pd.read_csv(features_path, header=None, dtype={0: np.int64})
    if raw.shape[1] < 3:
        raise ValueError(f"expected at least 3 fields in the features file, found {raw.shape[1]}")
    n_features = raw.shape[1] - 2  # minus txId and time step
    raw.columns = [TXID, TIMESTEP] + [f"feat_{i:03d}" for i in range(1, n_features + 1)]
    raw[TIMESTEP] = raw[TIMESTEP].astype(np.int16)
    for column in raw.columns[2:]:
        raw[column] = raw[column].astype(float_dtype)
    features = raw.set_index(TXID)

    logger.info("reading %s", edges_path.name)
    edges = pd.read_csv(edges_path, dtype=np.int64)
    if list(edges.columns) != ["txId1", "txId2"]:
        raise ValueError(f"unexpected edgelist columns: {list(edges.columns)}")

    logger.info("reading %s", classes_path.name)
    classes = pd.read_csv(classes_path, dtype={"txId": np.int64, "class": str})
    unexpected = set(classes["class"].unique()) - set(CLASS_MAP)
    if unexpected:
        raise ValueError(f"unexpected class values: {sorted(unexpected)}")
    labels = classes.set_index("txId")["class"].map(CLASS_MAP).astype(np.int8)

    return EllipticData(features=features, edges=edges, labels=labels)


def validate_elliptic(data: EllipticData) -> dict[str, Any]:
    """Compute and cross-check descriptive statistics of the Elliptic dataset.

    The temporal question -- whether edges connect transactions in the same time step -- is the
    one that matters most, because it determines whether a temporal split can leak structure
    across the split boundary. It is measured here rather than assumed either way.
    """
    features, edges, labels = data.features, data.edges, data.labels
    report: dict[str, Any] = {}

    report["n_nodes_features"] = int(len(features))
    report["n_nodes_labels"] = int(len(labels))
    report["n_edges"] = int(len(edges))
    report["n_features"] = len(data.feature_columns)

    # --- label coverage ---
    feature_ids = set(features.index)
    label_ids = set(labels.index)
    report["labels_without_features"] = len(label_ids - feature_ids)
    report["features_without_labels"] = len(feature_ids - label_ids)

    counts = labels.value_counts().to_dict()
    report["n_illicit"] = int(counts.get(ILLICIT, 0))
    report["n_licit"] = int(counts.get(LICIT, 0))
    report["n_unknown"] = int(counts.get(UNKNOWN, 0))
    labelled = report["n_illicit"] + report["n_licit"]
    report["n_labelled"] = labelled
    report["pct_labelled"] = round(100.0 * labelled / max(len(labels), 1), 3)
    report["pct_illicit_of_labelled"] = round(
        100.0 * report["n_illicit"] / max(labelled, 1), 3
    )
    report["imbalance_ratio_licit_per_illicit"] = round(
        report["n_licit"] / max(report["n_illicit"], 1), 2
    )

    # --- temporal structure ---
    steps = features[TIMESTEP]
    report["time_step_min"] = int(steps.min())
    report["time_step_max"] = int(steps.max())
    report["n_time_steps"] = int(steps.nunique())
    per_step = steps.value_counts().sort_index()
    report["nodes_per_step_min"] = int(per_step.min())
    report["nodes_per_step_max"] = int(per_step.max())
    report["nodes_per_step_mean"] = round(float(per_step.mean()), 1)

    # --- the leakage-critical question: do edges cross time steps? ---
    step_of = features[TIMESTEP]
    src = edges["txId1"].map(step_of)
    dst = edges["txId2"].map(step_of)
    known = src.notna() & dst.notna()
    report["edges_with_unknown_endpoint_step"] = int((~known).sum())
    same = (src[known] == dst[known])
    report["edges_within_same_step"] = int(same.sum())
    report["edges_crossing_steps"] = int((~same).sum())
    report["pct_edges_crossing_steps"] = round(
        100.0 * report["edges_crossing_steps"] / max(int(known.sum()), 1), 4
    )
    if report["edges_crossing_steps"]:
        gaps = (dst[known] - src[known]).abs()
        report["cross_step_gap_max"] = int(gaps.max())

    # --- per-step label distribution, for split design ---
    joined = pd.DataFrame({TIMESTEP: steps}).join(labels.rename("label"), how="left")
    joined["label"] = joined["label"].fillna(UNKNOWN).astype(np.int8)
    by_step = joined.groupby(TIMESTEP)["label"].agg(
        illicit=lambda s: int((s == ILLICIT).sum()),
        licit=lambda s: int((s == LICIT).sum()),
        unknown=lambda s: int((s == UNKNOWN).sum()),
    )
    report["steps_with_zero_illicit"] = [int(s) for s in by_step.index[by_step["illicit"] == 0]]
    report["per_step_label_counts"] = {
        int(step): {k: int(v) for k, v in row.items()} for step, row in by_step.iterrows()
    }

    # --- integrity checks ---
    report["duplicate_feature_rows"] = int(features.index.duplicated().sum())
    report["duplicate_edges"] = int(edges.duplicated().sum())
    report["self_loops"] = int((edges["txId1"] == edges["txId2"]).sum())
    report["nan_in_features"] = int(features[data.feature_columns].isna().sum().sum())

    return report


def load_wallets(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Load the Elliptic++ wallet (actor) feature table.

    Args:
        path: ``wallets_features.csv``.
        nrows: read only the first ``nrows`` rows, for a quick structural check on a 600 MB file.
    """
    return pd.read_csv(path, nrows=nrows)


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write a validation report as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("wrote %s", path)
