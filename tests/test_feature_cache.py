"""Tests for the feature cache key.

The cache is keyed by the extraction settings *and* by a hash of the source of every module that
produces a feature value. Without the second part, editing a feature implementation leaves the key
unchanged and a later run silently reuses vectors from the old code -- which happened once during
development and is exactly the kind of error that produces confidently wrong results.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from src.features.pipeline import ExtractionConfig, _feature_code_version


class TestCodeVersion:
    def test_is_a_stable_nonempty_hash(self):
        first = _feature_code_version()
        assert isinstance(first, str) and len(first) == 12
        assert first == _feature_code_version(), "must be deterministic within a process"

    def test_fingerprint_depends_on_the_code_version(self):
        config = ExtractionConfig()
        with mock.patch("src.features.pipeline._feature_code_version", return_value="a" * 12):
            first = config.fingerprint()
        with mock.patch("src.features.pipeline._feature_code_version", return_value="b" * 12):
            second = config.fingerprint()
        assert first != second, "a feature-code change must invalidate the cache"


class TestFingerprint:
    def test_identical_configs_agree(self):
        assert ExtractionConfig().fingerprint() == ExtractionConfig().fingerprint()

    def test_every_setting_changes_the_fingerprint(self):
        base = ExtractionConfig()
        variants = [
            ExtractionConfig(view="G_addr"),
            ExtractionConfig(radius=4),
            ExtractionConfig(filtration="temporal_sublevel"),
            ExtractionConfig(normalise="minmax"),
            ExtractionConfig(max_dim=2),
            ExtractionConfig(complex_dim=1),
            ExtractionConfig(max_nodes=1000),
            ExtractionConfig(resolution=64),
            ExtractionConfig(image_resolution=12),
            ExtractionConfig(num_landscapes=5),
            ExtractionConfig(families=("stats",)),
            ExtractionConfig(include_classical=False),
        ]
        fingerprints = {base.fingerprint()} | {v.fingerprint() for v in variants}
        assert len(fingerprints) == len(variants) + 1, "some setting does not affect the key"

    def test_cache_path_carries_the_readable_settings(self):
        path = ExtractionConfig(view="G_addr", radius=2,
                                filtration="temporal_sublevel").cache_path(Path("/tmp"))
        assert "G_addr" in path.name and "r2" in path.name and "temporal_sublevel" in path.name
        assert path.suffix == ".npz"


class TestTargetSetInCacheKey:
    """The extracted entity set belongs in the cache key.

    Without it, two runs sharing a configuration but extracting different entities -- an
    evaluation pool and a training reference sample -- map to the same file and overwrite each
    other on every alternation, each recomputing what the other just discarded. On the address
    graph that thrash costs tens of CPU-hours per cycle, and it is silent.
    """

    def test_different_target_sets_give_different_paths(self):
        import numpy as np

        config = ExtractionConfig(view="G_addr", radius=2)
        root = Path("/tmp")
        first = config.cache_path(root, np.array(["a", "b", "c"]))
        second = config.cache_path(root, np.array(["a", "b", "d"]))
        assert first != second

    def test_same_target_set_gives_a_stable_path(self):
        import numpy as np

        config = ExtractionConfig(view="G_addr", radius=2)
        root = Path("/tmp")
        targets = np.array(["a", "b", "c"])
        assert config.cache_path(root, targets) == config.cache_path(root, targets.copy())

    def test_target_order_is_part_of_the_key(self):
        """Row order must match, since the matrix is aligned to it positionally."""
        import numpy as np

        config = ExtractionConfig()
        root = Path("/tmp")
        assert (config.cache_path(root, np.array(["a", "b"]))
                != config.cache_path(root, np.array(["b", "a"])))

    def test_omitting_targets_reproduces_the_legacy_path(self):
        """Caches written before the target set entered the key must still be found."""
        import numpy as np

        config = ExtractionConfig(view="G_addr", radius=2)
        root = Path("/tmp")
        legacy = config.cache_path(root)
        assert "_t" not in legacy.stem.rsplit("_", 1)[-1]
        assert legacy != config.cache_path(root, np.array(["a"]))
