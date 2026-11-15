# Data

**No dataset is committed to this repository.** The Elliptic release permits research use but
does not grant redistribution, and the Elliptic++ repository carries no licence file at all, so
redistribution terms there are unspecified. `data/raw/` and `data/processed/` are git-ignored.
Run the download scripts to fetch from the official sources.

```powershell
.\.venv\Scripts\python.exe data\download_scripts\get_elliptic.py
.\.venv\Scripts\python.exe data\download_scripts\get_elliptic_plus_plus.py
```

Both scripts record source URL, byte size, SHA-256 and download time in
`data/raw/MANIFEST.json`. Downstream code refuses to read an unmanifested file, so every number
in the manuscript traces to a checksummed input.

## Sources

| Dataset | Source | Credential needed | Licence |
|---|---|---|---|
| Elliptic | `https://data.pyg.org/datasets/elliptic/` (the mirror `torch_geometric.datasets.EllipticBitcoinDataset` itself uses) | **none** | research use permitted; redistribution not granted |
| Elliptic++ | `https://github.com/git-disl/EllipticPlusPlus` (Git LFS) | **none**, but git-lfs must be installed | **unspecified** — no LICENSE file upstream |

Cite the dataset papers: Weber et al. (2019), arXiv:1908.02591 for Elliptic; Elmougy and Liu
(2023), `10.1145/3580305.3599803` for Elliptic++.

## Files fetched

Elliptic — `data/raw/elliptic/`:

| File | Size on disk |
|---|---|
| `elliptic_txs_features.csv` | 689,683,771 B |
| `elliptic_txs_edgelist.csv` | 4,470,584 B |
| `elliptic_txs_classes.csv` | 3,305,144 B |

Elliptic++ — `data/raw/elliptic_plus_plus/`:

| File | Size on disk |
|---|---|
| `Transactions Dataset/txs_features.csv` | 694,789,588 B |
| `Transactions Dataset/txs_edgelist.csv` | 4,470,584 B |
| `Transactions Dataset/txs_classes.csv` | 2,361,914 B |
| `Actors Dataset/wallets_features.csv` | 606,463,522 B |
| `Actors Dataset/wallets_classes.csv` | 30,421,134 B |
| `Actors Dataset/wallets_features_classes_combined.csv` | 609,000,048 B |
| `Actors Dataset/AddrAddr_edgelist.csv` | 200,631,481 B |
| `Actors Dataset/AddrTx_edgelist.csv` | 21,248,388 B |
| `Actors Dataset/TxAddr_edgelist.csv` | 36,702,878 B |

`Transactions Dataset/txs_edgelist.csv` is **byte-identical** to Elliptic's edgelist (same
SHA-256). The two datasets share one transaction graph; see `docs/DATA_FINDINGS.md`.

## Format notes that cause real mistakes

- `elliptic_txs_features.csv` has **no header row**. Field 0 is `txId`, field 1 is the time step,
  fields 2–166 are the anonymised features. Reading it with a header silently discards the first
  transaction.
- Elliptic labels are `1` / `2` / `unknown`; Elliptic++ writes the same thing as `1` / `2` / `3`.
  This project maps both to `1` (illicit) / `0` (licit) / `-1` (unlabelled).
- Elliptic++ CSVs are Git LFS objects. A failed `git lfs pull` leaves ~130 byte pointer stubs
  that parse as valid CSV. `src/utils/manifest.py` detects and rejects them.
- `wallets_features.csv` is keyed by **(address, time step)**, not by address: 1,268,260 rows over
  822,942 distinct addresses.

## Validation

```powershell
.\.venv\Scripts\python.exe scripts\validate_datasets.py
```

Writes `results/processed/dataset_validation.json` and `tables/table3_dataset_statistics.md`.
Findings that changed the research design are recorded in `docs/DATA_FINDINGS.md`.
