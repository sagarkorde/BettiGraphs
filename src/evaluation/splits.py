"""Temporal splits, frozen to match ``docs/PREREGISTRATION.md`` §3.

The split boundaries are module constants rather than parameters with defaults, because the
pre-registration commits to them and a function signature that invites changing them invites
exactly the post-hoc adjustment the pre-registration exists to prevent. A caller who genuinely
needs different boundaries must pass them explicitly and will show up in a diff.

The step-43 regime change sits inside the test window on purpose: illicit counts run 80-340 per
step through step 42 and collapse to 24, 24, 5, 2, 22 across steps 43-47, which is the shift
hypothesis H6 is about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Pre-registered primary split (PREREGISTRATION.md §3).
TRAIN_STEPS: tuple[int, int] = (1, 34)
VAL_STEPS: tuple[int, int] = (35, 41)
TEST_STEPS: tuple[int, int] = (42, 49)

#: Secondary split reported only for comparability with published numbers. No hypothesis is
#: tested on it.
LITERATURE_TRAIN: tuple[int, int] = (1, 34)
LITERATURE_TEST: tuple[int, int] = (35, 49)

#: The time step at which the documented regime change occurs, verified from the per-step
#: illicit counts in results/processed/dataset_validation.json.
REGIME_SHIFT_STEP: int = 43

ILLICIT, LICIT, UNKNOWN = 1, 0, -1


@dataclass(frozen=True)
class TemporalSplit:
    """Boolean masks over a time-step array, plus the counts needed to report honestly."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    time_steps: np.ndarray
    labels: np.ndarray

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for name, mask in (("train", self.train), ("val", self.val), ("test", self.test)):
            subset = self.labels[mask]
            out[name] = {
                "n_total": int(mask.sum()),
                "n_illicit": int((subset == ILLICIT).sum()),
                "n_licit": int((subset == LICIT).sum()),
                "n_unlabelled": int((subset == UNKNOWN).sum()),
            }
        return out

    def assert_disjoint(self) -> None:
        """Raise if the splits overlap or if their time-step ranges intersect.

        Called by every experiment before fitting. Overlapping splits are the single most
        damaging silent error available here, so this is an assertion rather than a warning.
        """
        if (self.train & self.val).any() or (self.train & self.test).any() or (self.val & self.test).any():
            raise AssertionError("temporal split masks overlap")
        train_steps = set(np.unique(self.time_steps[self.train]).tolist())
        val_steps = set(np.unique(self.time_steps[self.val]).tolist())
        test_steps = set(np.unique(self.time_steps[self.test]).tolist())
        if train_steps & val_steps or train_steps & test_steps or val_steps & test_steps:
            raise AssertionError(
                f"time-step ranges intersect: train={sorted(train_steps)[:3]}..., "
                f"val={sorted(val_steps)[:3]}..., test={sorted(test_steps)[:3]}..."
            )
        if train_steps and val_steps and max(train_steps) >= min(val_steps):
            raise AssertionError("train extends into or past the validation window")
        if val_steps and test_steps and max(val_steps) >= min(test_steps):
            raise AssertionError("validation extends into or past the test window")


def temporal_split(
    time_steps: np.ndarray,
    labels: np.ndarray,
    *,
    train: tuple[int, int] = TRAIN_STEPS,
    val: tuple[int, int] = VAL_STEPS,
    test: tuple[int, int] = TEST_STEPS,
    labelled_only: bool = True,
) -> TemporalSplit:
    """Build the pre-registered temporal split.

    Args:
        time_steps: time step of each row.
        labels: label of each row in ``{1, 0, -1}``.
        labelled_only: restrict the masks to labelled rows. Unlabelled nodes stay in the *graph*
            regardless -- removing them would sever the paths carrying the structure being
            measured -- but they cannot appear in a supervised split.

    Raises:
        ValueError: if a split window would contain no illicit examples, since PR-AUC on the
            illicit class is undefined there and silently reporting it would be misleading.
    """
    time_steps = np.asarray(time_steps)
    labels = np.asarray(labels)
    if time_steps.shape != labels.shape:
        raise ValueError(f"time_steps {time_steps.shape} and labels {labels.shape} disagree")

    usable = (labels != UNKNOWN) if labelled_only else np.ones_like(labels, dtype=bool)

    def window(bounds: tuple[int, int]) -> np.ndarray:
        low, high = bounds
        return usable & (time_steps >= low) & (time_steps <= high)

    split = TemporalSplit(
        train=window(train), val=window(val), test=window(test),
        time_steps=time_steps, labels=labels,
    )
    split.assert_disjoint()

    counts = split.counts()
    for name, stats in counts.items():
        if stats["n_illicit"] == 0:
            raise ValueError(
                f"the {name} window contains no illicit examples; PR-AUC on the illicit class "
                "would be undefined"
            )
        logger.info(
            "%-5s n=%d illicit=%d licit=%d (illicit %.2f%%)",
            name, stats["n_total"], stats["n_illicit"], stats["n_licit"],
            100.0 * stats["n_illicit"] / max(stats["n_total"], 1),
        )
    return split


def pre_post_shift_masks(
    time_steps: np.ndarray, mask: np.ndarray, *, shift_step: int = REGIME_SHIFT_STEP
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``mask`` into pre-shift and post-shift halves, for hypothesis H6."""
    return mask & (time_steps < shift_step), mask & (time_steps >= shift_step)


def per_step_frame(time_steps: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Per-time-step label counts.

    Every per-step metric in the manuscript is reported alongside these counts, because steps 45
    and 46 hold only 5 and 2 illicit transactions and a PR-AUC computed there is nearly
    meaningless in isolation.
    """
    frame = pd.DataFrame({"time_step": time_steps, "label": labels})
    return (
        frame.groupby("time_step")["label"]
        .agg(
            n_total="size",
            n_illicit=lambda s: int((s == ILLICIT).sum()),
            n_licit=lambda s: int((s == LICIT).sum()),
            n_unlabelled=lambda s: int((s == UNKNOWN).sum()),
        )
        .reset_index()
    )
