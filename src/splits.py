"""
splits.py — genome clustering + grouped train/calibration/test splits
Owner: Waji (pipeline)

This is the file that protects every number we report.

Near-identical genomes are everywhere in BV-BRC (outbreak isolates, resequenced
strains). A random row split puts copies of the same strain in both train and
test, and the model scores brilliantly by memorising strains rather than
learning resistance mechanisms. The brief calls this out explicitly and judges
look for it. So: cluster first, split by cluster, never by row.

Two entry points:
  cluster_genomes()  — FASTA -> cluster_id, via MinHash/Mash distance
  grouped_split()    — cluster_id -> train / calibration / test index sets

The calibration split is separate from train and test on purpose: Platt scaling
fitted on training data is optimistically biased, and fitting it on test data
leaks. calibration.py needs its own untouched slice.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Mash distance threshold for "same cluster".
#
# 0.05 Mash distance ~= 95% average nucleotide identity. This is deliberately
# CONSERVATIVE (it merges aggressively, producing fewer/larger clusters than a
# tighter threshold would). Erring toward over-merging costs us a little test-set
# size; erring the other way silently leaks near-identical genomes across the
# split and inflates every metric. We take the honest, pessimistic side.
#
# 95% ANI is also roughly the accepted species boundary, so within our single
# target species this groups strains, not species.
DEFAULT_MASH_THRESHOLD = 0.05

# MinHash sketch parameters. k=21 is the Mash default for bacterial genomes:
# long enough that random 21-mer collisions are negligible at a ~5 Mb genome
# size, short enough to stay sensitive to real similarity.
DEFAULT_KMER_SIZE = 21
DEFAULT_NUM_PERM = 256


class SplitError(ValueError):
    """Raised when a split would be statistically invalid."""


# --------------------------------------------------------------------------
# 1. Clustering: FASTA -> cluster_id
# --------------------------------------------------------------------------


def _read_fasta_sequence(path: Path) -> str:
    """Concatenate all contigs in a FASTA file into one uppercase sequence."""
    parts = []
    with open(path) as handle:
        for line in handle:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    return "".join(parts)


def _sketch(sequence: str, k: int, num_perm: int):
    """MinHash sketch of a sequence's k-mer set."""
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    for i in range(len(sequence) - k + 1):
        m.update(sequence[i : i + k].encode())
    return m


def mash_distance(jaccard: float, k: int = DEFAULT_KMER_SIZE) -> float:
    """
    Convert a Jaccard estimate to a Mash distance.

    Mash distance approximates the average nucleotide substitution rate:
        D = -(1/k) * ln( 2j / (1 + j) )
    A Jaccard of 1 gives D=0 (identical); a Jaccard of 0 gives D=1 (unrelated).
    """
    if jaccard <= 0.0:
        return 1.0
    if jaccard >= 1.0:
        return 0.0
    return -1.0 / k * math.log(2 * jaccard / (1 + jaccard))


def cluster_genomes(
    fasta_dir: str | Path,
    threshold: float = DEFAULT_MASH_THRESHOLD,
    k: int = DEFAULT_KMER_SIZE,
    num_perm: int = DEFAULT_NUM_PERM,
    pattern: str = "*.fna",
) -> pd.DataFrame:
    """
    Assign a cluster_id to every FASTA in a directory.

    Single-linkage clustering: two genomes join the same cluster if their Mash
    distance is below `threshold`, and clusters merge transitively. Single
    linkage is the cautious choice here — it errs toward merging, which errs
    toward NOT leaking across the split.

    Returns a DataFrame indexed by genome_id with a cluster_id column, ready to
    be joined into genomes.csv. genome_id is the FASTA filename stem.

    Note: this is O(n^2) in sketch comparisons. Fine for hackathon-scale
    datasets (a few thousand genomes); swap in Mash/sourmash proper if the real
    dataset is much larger.
    """
    fasta_dir = Path(fasta_dir)
    paths = sorted(p for ext in (pattern, "*.fasta", "*.fa") for p in fasta_dir.glob(ext))
    if not paths:
        raise SplitError(f"No FASTA files found in {fasta_dir}")

    genome_ids = [p.stem for p in paths]
    sketches = [_sketch(_read_fasta_sequence(p), k, num_perm) for p in paths]

    # Union-find for single-linkage clustering.
    parent = list(range(len(genome_ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(sketches)):
        for j in range(i + 1, len(sketches)):
            if mash_distance(sketches[i].jaccard(sketches[j]), k) < threshold:
                union(i, j)

    # Relabel roots as stable, human-readable CL-#### ids.
    roots = sorted({find(i) for i in range(len(genome_ids))})
    label = {root: f"CL-{n:04d}" for n, root in enumerate(roots)}

    return pd.DataFrame(
        {"cluster_id": [label[find(i)] for i in range(len(genome_ids))]},
        index=pd.Index(genome_ids, name="genome_id"),
    )


# --------------------------------------------------------------------------
# 2. Splitting: cluster_id -> train / calibration / test
# --------------------------------------------------------------------------


@dataclass
class GroupedSplit:
    """Row positions for each split, guaranteed cluster-disjoint."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "calibration": len(self.calibration),
            "test": len(self.test),
        }


def grouped_split(
    groups: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series | None = None,
    test_size: float = 0.2,
    calibration_size: float = 0.15,
    seed: int = 0,
) -> GroupedSplit:
    """
    Split row positions into train / calibration / test with no cluster spanning
    two splits.

    Clusters are allocated whole. When `y` is supplied, clusters are dealt out
    largest-first to whichever split is furthest below its target size — this
    keeps the small splits from collapsing to a single huge cluster, which is
    the usual failure mode when a handful of clusters dominate the dataset.

    `groups` and `y` must be aligned row-for-row with the feature matrix, i.e.
    exactly what Dataset.xy_for_drug() hands back.
    """
    groups = np.asarray(groups)
    n = len(groups)
    if n == 0:
        raise SplitError("Cannot split an empty dataset")
    if not 0 < test_size + calibration_size < 1:
        raise SplitError(
            f"test_size + calibration_size must be in (0, 1); "
            f"got {test_size} + {calibration_size}"
        )

    members: dict[str, list[int]] = defaultdict(list)
    for pos, g in enumerate(groups):
        members[g].append(pos)

    if len(members) < 3:
        raise SplitError(
            f"Only {len(members)} cluster(s) present — cannot build three disjoint "
            "splits. Either the data is too small or clustering over-merged; "
            "check the Mash threshold before trusting any metric from this run."
        )

    targets = {
        "train": (1 - test_size - calibration_size) * n,
        "calibration": calibration_size * n,
        "test": test_size * n,
    }
    assigned: dict[str, list[int]] = {k: [] for k in targets}

    # Largest clusters first: place the constrained items while there is still
    # room to place them.
    rng = np.random.default_rng(seed)
    order = sorted(members, key=lambda g: (-len(members[g]), g))
    # Break size ties randomly so the split is not an artefact of cluster naming.
    order = sorted(order, key=lambda g: (-len(members[g]), rng.random()))

    for g in order:
        deficits = {k: targets[k] - len(assigned[k]) for k in targets}
        pick = max(deficits, key=deficits.get)
        assigned[pick].extend(members[g])

    split = GroupedSplit(
        train=np.sort(np.array(assigned["train"], dtype=int)),
        calibration=np.sort(np.array(assigned["calibration"], dtype=int)),
        test=np.sort(np.array(assigned["test"], dtype=int)),
    )
    _verify(split, groups, y)
    return split


def _verify(split: GroupedSplit, groups: np.ndarray, y=None) -> None:
    """Fail loudly rather than report a leaked or degenerate split."""
    parts = {"train": split.train, "calibration": split.calibration, "test": split.test}

    for name, idx in parts.items():
        if len(idx) == 0:
            raise SplitError(
                f"The {name} split is empty. With few clusters this happens easily — "
                "reduce calibration_size/test_size or gather more clusters."
            )

    # The whole point of this module: no cluster may appear in two splits.
    seen: dict[str, str] = {}
    for name, idx in parts.items():
        for g in set(groups[idx]):
            if g in seen:
                raise SplitError(
                    f"Cluster {g!r} appears in both {seen[g]} and {name} — "
                    "this is exactly the leak grouped splitting exists to prevent."
                )
            seen[g] = name

    if y is not None:
        y = np.asarray(y)
        for name, idx in parts.items():
            classes = set(np.unique(y[idx]))
            if len(classes) < 2:
                raise SplitError(
                    f"The {name} split contains only class {classes} — metrics like "
                    "AUROC are undefined and calibration cannot be fitted. This drug "
                    "should be reported as insufficient data, not modelled."
                )


def split_report(split: GroupedSplit, groups: np.ndarray, y=None) -> pd.DataFrame:
    """Per-split row/cluster counts and class balance — paste into the writeup."""
    groups = np.asarray(groups)
    rows = []
    for name, idx in (
        ("train", split.train),
        ("calibration", split.calibration),
        ("test", split.test),
    ):
        row = {
            "split": name,
            "n_genomes": len(idx),
            "n_clusters": len(set(groups[idx])),
        }
        if y is not None:
            yi = np.asarray(y)[idx]
            row["n_resistant"] = int(yi.sum())
            row["pct_resistant"] = round(100 * float(yi.mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_io import load_dataset

    ds = load_dataset(Path(__file__).parent.parent / "data" / "synthetic")
    for drug in ds.drugs:
        X, y, groups = ds.xy_for_drug(drug)
        try:
            split = grouped_split(groups, y)
        except SplitError as exc:
            print(f"\n{drug}: SKIPPED — {exc}")
            continue
        print(f"\n{drug}")
        print(split_report(split, groups, y).to_string(index=False))
