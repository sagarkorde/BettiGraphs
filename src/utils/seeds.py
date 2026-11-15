"""Seeding and run provenance.

Every experiment in this project runs under an explicit seed and records it alongside its
output, so that a reported mean +/- standard deviation can be traced to the exact ten runs that
produced it.

On GPU determinism: exact bit-for-bit reproducibility is not claimed. cuDNN chooses
non-deterministic kernels for several operations, and forcing deterministic algorithms changes
runtime enough to distort the scalability measurements this project reports. Conclusions
therefore rest on paired significance tests across seeds, not on single-run point values.
:func:`set_all_seeds` accepts ``deterministic=True`` for callers who want the stricter setting.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: The twenty seeds used for every multi-seed experiment. Fixed here, not drawn at runtime, so
#: that "20 seeds" means the same twenty everywhere in the paper.
#:
#: Raised from ten at the author's direction. The reason is statistical rather than cosmetic: the
#: paired signed-rank test is discrete with a two-sided floor of 2/2**n, which at ten seeds is
#: 0.00195 -- above Holm's strictest threshold over a 31-comparison family (0.05/31 = 0.00161), so
#: no comparison could reach significance at any effect size. At twenty seeds the floor is 1.9e-6
#: and the originally frozen rule works, which makes pre-registration Amendment 4 unnecessary
#: rather than adding a fifth amendment.
#:
#: The first ten are the original ten, so results at either count are directly comparable.
SEEDS: tuple[int, ...] = tuple(range(20))


def set_all_seeds(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if present) torch.

    Args:
        seed: the seed value.
        deterministic: if True, additionally request deterministic cuDNN algorithms. Slower,
            and it changes the timings reported in the scalability analysis, so the default is
            False and variance is handled by averaging over :data:`SEEDS` instead.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:  # pragma: no cover - torch version dependent
            logger.warning("could not enable deterministic algorithms: %s", exc)
    else:
        torch.backends.cudnn.benchmark = True


@dataclass
class RunProvenance:
    """Environment fingerprint stamped into every result file."""

    seed: int
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform_: str = field(default_factory=platform.platform)
    numpy: str = field(default_factory=lambda: np.__version__)
    torch: str | None = None
    cuda: str | None = None
    gpu: str | None = None
    git_commit: str | None = None

    def __post_init__(self) -> None:
        try:
            import torch

            self.torch = torch.__version__
            self.cuda = torch.version.cuda
            if torch.cuda.is_available():
                self.gpu = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        self.git_commit = _git_commit()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_commit() -> str | None:
    """Current HEAD short hash, or None outside a repository."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
