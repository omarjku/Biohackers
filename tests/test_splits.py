"""
Run with: pytest tests/test_splits.py -v

These tests exist because a broken allocator does not crash — it quietly returns
a split that looks fine and silently invalidates every metric downstream.

The regression actually pinned here is the squared-deviation cost function: it
rewards HITTING a per-split target, so one cluster fills a small split exactly
and the small splits get served first. Measured, that drove train from 65% of
rows down to 34% while leaving cluster-disjointness perfectly intact — the leak
test would never have caught it. Hence test_split_sizes_match_requested_
proportions.

(An earlier version of this docstring also claimed that using absolute rather
than relative class deficits starves the small splits. That was asserted without
measurement and is false: the two are equivalent on this data. The comment in
splits.py records the real, weaker reason for the normalization.)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_io import load_dataset
from splits import SplitError, grouped_split, mash_distance

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"
SEEDS = range(5)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA_DIR)


def _synthetic_groups(n_clusters=20, per_cluster=10, resistant_clusters=8, seed=0):
    """
    Clusters that are phenotypically homogeneous — every genome in a cluster
    shares a label. This is the hard case for a grouped split and the realistic
    one: outbreak isolates of a resistant strain cluster together.
    """
    rng = np.random.default_rng(seed)
    groups, y = [], []
    for c in range(n_clusters):
        groups += [f"CL-{c:03d}"] * per_cluster
        y += [1 if c < resistant_clusters else 0] * per_cluster
    order = rng.permutation(len(groups))
    return np.array(groups)[order], np.array(y)[order]


# --------------------------------------------------------------------------
# The invariant the whole module exists for
# --------------------------------------------------------------------------


def test_no_cluster_spans_two_splits(dataset):
    for drug in dataset.drugs:
        _, y, groups = dataset.xy_for_drug(drug)
        for seed in SEEDS:
            split = grouped_split(groups, y=y, seed=seed)
            parts = {
                "train": set(groups[split.train]),
                "calibration": set(groups[split.calibration]),
                "test": set(groups[split.test]),
            }
            assert not parts["train"] & parts["test"], f"{drug} seed={seed}: train/test leak"
            assert not parts["train"] & parts["calibration"]
            assert not parts["calibration"] & parts["test"]


def test_every_row_assigned_exactly_once(dataset):
    for drug in dataset.drugs:
        _, y, groups = dataset.xy_for_drug(drug)
        split = grouped_split(groups, y=y, seed=0)
        allocated = np.concatenate([split.train, split.calibration, split.test])
        assert sorted(allocated) == list(range(len(groups)))


# --------------------------------------------------------------------------
# Regression: split sizes must track the requested proportions
# --------------------------------------------------------------------------


def test_split_sizes_match_requested_proportions(dataset):
    """Guards the squared-deviation bug, which drove train from 65% to 34%."""
    for drug in dataset.drugs:
        _, y, groups = dataset.xy_for_drug(drug)
        for seed in SEEDS:
            split = grouped_split(groups, y=y, test_size=0.2, calibration_size=0.15, seed=seed)
            n = len(groups)
            fractions = {
                "train": len(split.train) / n,
                "calibration": len(split.calibration) / n,
                "test": len(split.test) / n,
            }
            # Clusters are indivisible, so exact proportions are impossible;
            # 10 points of slack is generous but still catches a collapse.
            assert abs(fractions["train"] - 0.65) < 0.10, f"{drug} seed={seed}: {fractions}"
            assert abs(fractions["calibration"] - 0.15) < 0.10, f"{drug} seed={seed}: {fractions}"
            assert abs(fractions["test"] - 0.20) < 0.10, f"{drug} seed={seed}: {fractions}"


def test_train_is_the_largest_split(dataset):
    """Cheap ordering sanity check on top of the proportion assertions above."""
    for drug in dataset.drugs:
        _, y, groups = dataset.xy_for_drug(drug)
        split = grouped_split(groups, y=y, seed=0)
        assert len(split.train) > len(split.test) > len(split.calibration)


# --------------------------------------------------------------------------
# Regression: label balance, the thing the allocator was changed to fix
# --------------------------------------------------------------------------


def test_prevalence_is_comparable_across_splits(dataset):
    """
    Resistance prevalence must not drift far between splits. Before the allocator
    was label-aware, ciprofloxacin sat at 40% resistant in train against 70% in
    test — which made Platt scaling fitted on one prevalence wrong for the other,
    and inflated the apparent cost of grouped splitting.
    """
    for drug in dataset.drugs:
        _, y, groups = dataset.xy_for_drug(drug)
        for seed in SEEDS:
            split = grouped_split(groups, y=y, seed=seed)
            rates = [
                y[split.train].mean(),
                y[split.calibration].mean(),
                y[split.test].mean(),
            ]
            spread = max(rates) - min(rates)
            assert spread < 0.15, (
                f"{drug} seed={seed}: prevalence spread {spread:.2f} across splits "
                f"(train/cal/test = {[round(r, 3) for r in rates]})"
            )


def test_label_aware_beats_size_only_on_homogeneous_clusters():
    """
    The direct comparison, on clusters engineered to be phenotypically pure.
    Passing y must produce a tighter prevalence spread than omitting it.
    """
    groups, y = _synthetic_groups()

    with_labels = grouped_split(groups, y=y, seed=0)
    without_labels = grouped_split(groups, y=None, seed=0)

    def spread(split):
        rates = [y[split.train].mean(), y[split.calibration].mean(), y[split.test].mean()]
        return max(rates) - min(rates)

    assert spread(with_labels) < spread(without_labels)


# --------------------------------------------------------------------------
# Failing loudly instead of returning something unusable
# --------------------------------------------------------------------------


def test_rejects_too_few_clusters():
    groups = np.array(["CL-000"] * 10 + ["CL-001"] * 10)
    y = np.array([1] * 10 + [0] * 10)
    with pytest.raises(SplitError, match="cluster"):
        grouped_split(groups, y=y)


def test_rejects_empty_dataset():
    with pytest.raises(SplitError, match="empty"):
        grouped_split(np.array([]), y=np.array([]))


def test_rejects_single_class_split():
    """A split holding one class cannot be scored or calibrated — say so."""
    groups, y = _synthetic_groups(n_clusters=6, per_cluster=5, resistant_clusters=1)
    with pytest.raises(SplitError, match="only class"):
        grouped_split(groups, y=y)


def test_rejects_impossible_proportions():
    groups, y = _synthetic_groups()
    with pytest.raises(SplitError, match="must be in"):
        grouped_split(groups, y=y, test_size=0.7, calibration_size=0.5)


def test_seed_is_deterministic(dataset):
    _, y, groups = dataset.xy_for_drug(dataset.drugs[0])
    a = grouped_split(groups, y=y, seed=3)
    b = grouped_split(groups, y=y, seed=3)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)


# --------------------------------------------------------------------------
# Mash distance
# --------------------------------------------------------------------------


def test_mash_distance_endpoints():
    assert mash_distance(1.0) == 0.0     # identical
    assert mash_distance(0.0) == 1.0     # nothing in common


def test_mash_distance_is_monotonic():
    """More shared k-mers must never mean a larger distance."""
    jaccards = [0.05, 0.2, 0.5, 0.8, 0.95]
    distances = [mash_distance(j) for j in jaccards]
    assert distances == sorted(distances, reverse=True)


def test_mash_threshold_matches_ani_intuition():
    """~95% ANI is the documented 0.05 threshold; a 90%-Jaccard pair sits inside it."""
    assert mash_distance(0.90) < 0.05
