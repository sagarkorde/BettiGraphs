# Computational environment

Recorded for reproducibility. Every runtime figure in the manuscript's scalability analysis was
measured on this machine; a different machine will produce different absolute timings, so the
paper reports relative scaling behaviour alongside the absolute numbers.

## Hardware

| Component | Value |
|---|---|
| CPU | Intel 64 Family 6 Model 183 Stepping 1 (Raptor Lake-HX class), **32 logical cores** |
| RAM | 63.7 GiB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GiB VRAM |
| NVIDIA driver | 610.88 |
| Disk | project on `E:` (NTFS), 103 GB free at setup |

## Software

| Component | Value |
|---|---|
| OS | Windows 11 Pro, `Windows-10-10.0.26200-SP0` (build 26200) |
| Interpreter | CPython **3.10.10** (tags/v3.10.10:aad5f6a, 2023-02-07) [MSC v.1929 64-bit] |
| Interpreter path | `C:\Program Files\Python310\python.exe` |
| Virtual environment | `.venv`, created with `--system-site-packages` |
| CUDA (torch build) | 12.8 |
| `torch.cuda.is_available()` | `True` |
| git | 2.51.2.windows.1 |
| git-lfs | 3.7.1 (required for Elliptic++) |

**Python 3.10 is a hard requirement, not a preference.** The machine's default `python` is 3.14.5
and has no scientific stack at all; `gudhi`, `ripser` and the `torch` cu128 wheels publish no
3.14 builds. All commands must go through `.venv\Scripts\python.exe` or `py -3.10`.

The virtual environment inherits the system site-packages so that the existing CUDA-enabled
`torch` build is reused rather than re-downloaded. Only four packages are installed into
`.venv` itself:

```
gudhi==3.13.0  persim==0.3.8  ripser==0.6.15  POT==0.9.7.post1
```

All four resolved to prebuilt `cp310-win_amd64` wheels, so no C++ toolchain is required.

## Dependency files

| File | Contents |
|---|---|
| `requirements.txt` | direct dependencies, pinned, with rationale |
| `requirements-freeze.txt` | full transitive closure (265 entries) as installed |
| `environment.yml` | conda equivalent |

Three packages are deliberately **not** dependencies:

- `shap` — XGBoost and LightGBM expose exact TreeSHAP natively (`pred_contribs=True`,
  `pred_contrib=True`), so the interpretability analysis needs no extra library.
- `statsmodels` — Holm–Bonferroni correction is implemented in
  `src/evaluation/statistics.py`; SciPy covers the tests themselves.
- `giotto-tda` — it pins `scikit-learn` and `numpy` versions, and its vectorisers fit their
  ranges to the data they see, which is a leakage hazard across a train/test boundary.
  `src/features/topological.py` replaces it with fixed-range implementations.

## Determinism

Seeds are set for `random`, `numpy`, `torch` (CPU and CUDA) and scikit-learn estimators via
`src/utils/seeds.py`. Every experiment records the seed it ran under in its output.

Exact bit-for-bit reproducibility of GPU results is **not** claimed: cuDNN selects
non-deterministic kernels for some operations, and forcing deterministic algorithms changes
runtime enough to distort the scalability measurements. This is why every reported result is a
mean ± standard deviation over 10 seeds rather than a single run, and why conclusions rest on
paired significance tests rather than point differences.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The topology suite contains known-answer homology tests — a 5-cycle must give β₁ = 1, the
octahedron's clique complex must give β₂ = 1 as a 2-sphere, a complete graph's clique complex
must be contractible, and β₁ of a 1-skeleton must equal its cycle rank — so a regression in the
persistence layer fails loudly instead of producing plausible numbers.
