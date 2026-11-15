"""Gate 3: validate both datasets and emit the dataset-statistics table.

Reads only checksummed files recorded in ``data/raw/MANIFEST.json``, computes every descriptive
statistic from the files themselves, and writes:

* ``results/processed/dataset_validation.json`` -- the full report
* ``tables/table3_dataset_statistics.md`` -- Table 3 of the manuscript

Nothing here touches labels in relation to features, so it can precede the frozen
pre-registration without compromising it: knowing the cohort size and temporal layout is a
prerequisite for specifying a temporal split, not a peek at the answer.

Usage::

    .venv\\Scripts\\python.exe scripts\\validate_datasets.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.elliptic import load_elliptic, validate_elliptic, write_report  # noqa: E402
from src.utils.manifest import load as load_manifest  # noqa: E402

RAW = REPO_ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.json"
OUT_JSON = REPO_ROOT / "results" / "processed" / "dataset_validation.json"
OUT_TABLE = REPO_ROOT / "tables" / "table3_dataset_statistics.md"

logger = logging.getLogger("validate")


def _require_manifested(keys: list[str]) -> None:
    """Refuse to proceed on data that was not recorded by a download script."""
    manifest = load_manifest(MANIFEST)
    missing = [k for k in keys if k not in manifest["artifacts"]]
    if missing:
        raise RuntimeError(
            f"these artefacts are not in the manifest: {missing}. Every number in the "
            "manuscript must trace to a checksummed input, so unmanifested files are refused."
        )


def validate_elliptic_pp(raw_dir: Path) -> dict:
    """Structural validation of the Elliptic++ tables, read in chunks where they are large."""
    report: dict = {}
    actors = raw_dir / "Actors Dataset"
    txs = raw_dir / "Transactions Dataset"

    # --- wallet labels: small enough to read whole ---
    wallet_classes = pd.read_csv(actors / "wallets_classes.csv")
    report["wallet_classes_columns"] = list(wallet_classes.columns)
    report["n_wallet_label_rows"] = int(len(wallet_classes))
    label_col = "class" if "class" in wallet_classes.columns else wallet_classes.columns[-1]
    counts = wallet_classes[label_col].value_counts().to_dict()
    report["wallet_class_distribution"] = {str(k): int(v) for k, v in counts.items()}
    addr_col = wallet_classes.columns[0]
    report["n_unique_wallet_ids"] = int(wallet_classes[addr_col].nunique())

    # --- address-address graph: 200 MB, count rows and unique endpoints in chunks ---
    addr_addr = actors / "AddrAddr_edgelist.csv"
    n_rows = 0
    sources: set = set()
    targets: set = set()
    header: list[str] = []
    for chunk in pd.read_csv(addr_addr, chunksize=2_000_000):
        if not header:
            header = list(chunk.columns)
        n_rows += len(chunk)
        sources.update(chunk.iloc[:, 0].unique().tolist())
        targets.update(chunk.iloc[:, 1].unique().tolist())
    report["addr_addr_columns"] = header
    report["addr_addr_n_edges"] = n_rows
    report["addr_addr_n_unique_sources"] = len(sources)
    report["addr_addr_n_unique_targets"] = len(targets)
    report["addr_addr_n_unique_addresses"] = len(sources | targets)

    # --- bipartite address-transaction edges ---
    for name, key in (("AddrTx_edgelist.csv", "addr_tx"), ("TxAddr_edgelist.csv", "tx_addr")):
        total = 0
        cols: list[str] = []
        for chunk in pd.read_csv(actors / name, chunksize=2_000_000):
            if not cols:
                cols = list(chunk.columns)
            total += len(chunk)
        report[f"{key}_columns"] = cols
        report[f"{key}_n_edges"] = total

    # --- wallet feature table: header and row count only ---
    wallets_head = pd.read_csv(actors / "wallets_features.csv", nrows=5)
    report["wallet_feature_columns"] = list(wallets_head.columns)
    report["n_wallet_features"] = len(wallets_head.columns)
    total = 0
    for chunk in pd.read_csv(actors / "wallets_features.csv", usecols=[0], chunksize=2_000_000):
        total += len(chunk)
    report["n_wallet_feature_rows"] = total

    # --- transaction tables, for comparison against plain Elliptic ---
    txs_head = pd.read_csv(txs / "txs_features.csv", nrows=5)
    report["pp_txs_feature_columns_count"] = len(txs_head.columns)
    report["pp_txs_feature_columns_sample"] = list(txs_head.columns)[:8]
    pp_classes = pd.read_csv(txs / "txs_classes.csv")
    report["pp_txs_class_columns"] = list(pp_classes.columns)
    report["pp_txs_n_label_rows"] = int(len(pp_classes))
    pp_label_col = "class" if "class" in pp_classes.columns else pp_classes.columns[-1]
    report["pp_txs_class_distribution"] = {
        str(k): int(v) for k, v in pp_classes[pp_label_col].value_counts().to_dict().items()
    }
    return report


def build_table3(elliptic: dict, pp: dict) -> str:
    """Render Table 3 from the validation reports. Every cell is a computed value."""
    lines = [
        "# Table 3 — Dataset statistics",
        "",
        "Every value is computed by `scripts/validate_datasets.py` from the checksummed files",
        "recorded in `data/raw/MANIFEST.json`. No figure is taken from the literature.",
        "",
        "## Elliptic (transaction-level)",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Transactions (nodes) | {elliptic['n_nodes_features']:,} |",
        f"| Directed edges | {elliptic['n_edges']:,} |",
        f"| Features per transaction | {elliptic['n_features']:,} |",
        f"| Time steps | {elliptic['n_time_steps']} "
        f"({elliptic['time_step_min']}–{elliptic['time_step_max']}) |",
        f"| Nodes per time step | min {elliptic['nodes_per_step_min']:,}, "
        f"mean {elliptic['nodes_per_step_mean']:,}, max {elliptic['nodes_per_step_max']:,} |",
        f"| Illicit (class 1) | {elliptic['n_illicit']:,} |",
        f"| Licit (class 2) | {elliptic['n_licit']:,} |",
        f"| Unlabelled | {elliptic['n_unknown']:,} |",
        f"| Labelled share | {elliptic['pct_labelled']}% |",
        f"| Illicit share of labelled | {elliptic['pct_illicit_of_labelled']}% |",
        f"| Imbalance (licit : illicit) | {elliptic['imbalance_ratio_licit_per_illicit']} : 1 |",
        f"| Edges within one time step | {elliptic['edges_within_same_step']:,} |",
        f"| Edges crossing time steps | {elliptic['edges_crossing_steps']:,} "
        f"({elliptic['pct_edges_crossing_steps']}%) |",
        f"| Duplicate edges | {elliptic['duplicate_edges']:,} |",
        f"| Self-loops | {elliptic['self_loops']:,} |",
        f"| Missing feature values | {elliptic['nan_in_features']:,} |",
        "",
        "## Elliptic++ (adds address / wallet level)",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Wallet label rows | {pp['n_wallet_label_rows']:,} |",
        f"| Unique wallet identifiers | {pp['n_unique_wallet_ids']:,} |",
        f"| Wallet feature columns | {pp['n_wallet_features']:,} |",
        f"| Wallet feature rows | {pp['n_wallet_feature_rows']:,} |",
        f"| Address–address edges | {pp['addr_addr_n_edges']:,} |",
        f"| Unique addresses in that graph | {pp['addr_addr_n_unique_addresses']:,} |",
        f"| Address→transaction edges | {pp['addr_tx_n_edges']:,} |",
        f"| Transaction→address edges | {pp['tx_addr_n_edges']:,} |",
        f"| Transaction label rows | {pp['pp_txs_n_label_rows']:,} |",
        f"| Transaction feature columns | {pp['pp_txs_feature_columns_count']:,} |",
        "",
        "### Wallet class distribution",
        "",
        "| Class | Count |",
        "|---|---|",
    ]
    for key, value in sorted(pp["wallet_class_distribution"].items()):
        lines.append(f"| {key} | {value:,} |")
    lines += [
        "",
        "### Elliptic++ transaction class distribution",
        "",
        "| Class | Count |",
        "|---|---|",
    ]
    for key, value in sorted(pp["pp_txs_class_distribution"].items()):
        lines.append(f"| {key} | {value:,} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    _require_manifested(
        [
            "elliptic/elliptic_txs_features.csv",
            "elliptic/elliptic_txs_edgelist.csv",
            "elliptic/elliptic_txs_classes.csv",
            "elliptic_pp/wallets_classes",
            "elliptic_pp/addr_addr_edgelist",
        ]
    )

    logger.info("validating Elliptic")
    data = load_elliptic(RAW / "elliptic")
    elliptic_report = validate_elliptic(data)

    logger.info("validating Elliptic++")
    pp_report = validate_elliptic_pp(RAW / "elliptic_plus_plus")

    report = {"elliptic": elliptic_report, "elliptic_pp": pp_report}
    write_report(report, OUT_JSON)

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(build_table3(elliptic_report, pp_report), encoding="utf-8")
    logger.info("wrote %s", OUT_TABLE)

    summary = {k: v for k, v in elliptic_report.items() if k != "per_step_label_counts"}
    print(json.dumps({"elliptic": summary, "elliptic_pp": pp_report}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
