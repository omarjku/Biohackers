"""
evaluation.py — offline metrics + plots for the demo/presentation
Owner: was unassigned; implemented by Waji alongside the pipeline. Hand it over
if someone picks up evaluation properly.

Everything here evaluates the SHIPPING code path — predictor -> calibration ->
no-call gate — on the held-out TEST slice of the grouped split. It deliberately
does not re-derive probabilities by hand: if the production path has a bug, these
numbers should show it, not paper over it.

Two ideas the metric layout depends on:

1. Probabilistic metrics (AUROC, PR-AUC, Brier) are computed on EVERY test row,
   because they score the underlying probability and are unaffected by whether we
   chose to answer. Decision metrics (balanced accuracy, recall_R, recall_S, F1)
   are computed only on rows we actually answered — no-call and not_applicable
   rows are excluded and reported separately as coverage. Mixing the two lets a
   model look good by abstaining on everything hard, so coverage is always
   printed next to the accuracy it bought.

2. PR-AUC uses RESISTANT as the positive class (y=1, see data_io). Under class
   imbalance AUROC flatters a model that ranks the plentiful susceptible genomes
   well; PR-AUC on the resistant class reflects whether we catch the cases that
   matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from calibration import NO_CALL_HIGH, NO_CALL_LOW, Calibrator, predict_calibrated
from data_io import Dataset
from predictor import Predictor
from splits import GroupedSplit, grouped_split

# Bins for the reliability diagram. 10 is conventional; with a few hundred test
# genomes per drug, more bins produce noise that looks like miscalibration.
N_RELIABILITY_BINS = 10

BANNER = (
    "SYNTHETIC DATA — these numbers describe fixtures, not real E. coli. "
    "Not a benchmark."
)

REAL_DATA_BANNER = (
    "REAL DATA (E. coli, BV-BRC) — research prototype. "
    "All results must be confirmed with standard laboratory testing."
)


def banner_for(data_dir: Path) -> str:
    """
    Pick the provenance banner from the data actually loaded.

    This was a single hardcoded SYNTHETIC constant, stamped onto every figure and
    printed at the top of every run. The first real-data run therefore produced a
    dashboard labelled SYNTHETIC — a mislabelled plot is the one output that
    survives into a slide deck without its context, so provenance is derived
    here, never assumed.
    """
    return BANNER if "synthetic" in Path(data_dir).name.lower() else REAL_DATA_BANNER


@dataclass
class DrugEvaluation:
    """Everything measured for one antibiotic on its own test slice."""

    drug: str
    y_true: np.ndarray          # all test rows
    y_prob: np.ndarray          # calibrated P(resistant), all test rows
    y_prob_raw: np.ndarray      # uncalibrated, for the before/after comparison
    calls: list[str]            # production call per test row
    clusters: np.ndarray        # cluster_id per test row

    @property
    def answered(self) -> np.ndarray:
        """Mask of rows where the pipeline committed to work/fail."""
        return np.array([c in ("likely_to_work", "likely_to_fail") for c in self.calls])

    @property
    def y_pred(self) -> np.ndarray:
        """Decisions as 0/1, only meaningful under the `answered` mask."""
        return np.array([1 if c == "likely_to_fail" else 0 for c in self.calls])


def _safe(fn, *args, **kwargs) -> float:
    """Metrics are undefined on single-class slices; report NaN, never crash."""
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def _decision_metrics_defined(y_true_answered: np.ndarray) -> bool:
    """
    Are per-class decision metrics meaningful on the rows we actually answered?

    Only when both classes survive the no-call gate. sklearn does NOT raise on a
    single-class y_true — balanced_accuracy_score quietly degrades to plain
    accuracy, and recall for the absent class returns 0 via zero_division. That
    combination produced a genuinely misleading row: ampicillin answers only
    genomes that are truly resistant, so it reported bal_acc 0.917 alongside
    recall_R 1.000 and recall_S 0.000 — three numbers that cannot all be true.
    The 0.917 was accuracy on a single class wearing balanced accuracy's name.

    Suppressing to NaN makes the gap visible: coverage still reports, and the
    seed count in multi_seed_metrics shows how often a drug was scoreable at all.
    """
    return len(np.unique(y_true_answered)) >= 2


# --------------------------------------------------------------------------
# 1. Run the real pipeline over the test slice
# --------------------------------------------------------------------------


def evaluate_drug(
    dataset: Dataset,
    predictor: Predictor,
    calibrator: Calibrator,
    drug: str,
) -> DrugEvaluation:
    """Push every test genome through predictor -> calibration -> no-call gate."""
    split = predictor.splits[drug]
    model = predictor.models[drug]

    X, y, groups = dataset.xy_for_drug(drug)
    X_test = X.iloc[split.test]
    y_test = np.asarray(y)[split.test]
    clusters = np.asarray(groups)[split.test]

    raw, probs, calls = [], [], []
    for genome_id, row in X_test.iterrows():
        raw.append(model.probability_resistant(row))
        # Use the public path so we evaluate exactly what the app would show.
        prediction = predict_calibrated(
            dataset, predictor, calibrator, genome_id, drugs=[drug]
        )[0]
        probs.append(prediction.confidence)
        calls.append(prediction.call)

    return DrugEvaluation(
        drug=drug,
        y_true=y_test,
        y_prob=np.array(probs),
        y_prob_raw=np.array(raw),
        calls=calls,
        clusters=clusters,
    )


def metrics_table(evaluations: list[DrugEvaluation]) -> pd.DataFrame:
    """The per-drug table for the writeup. One row per antibiotic."""
    rows = []
    for ev in evaluations:
        mask = ev.answered
        yt_all, yp_all = ev.y_true, ev.y_prob
        yt, yp = ev.y_true[mask], ev.y_pred[mask]

        # Decision metrics are only defined when both classes survive the no-call
        # gate; otherwise they are NaN rather than a flattering artefact.
        nan = float("nan")
        defined = _decision_metrics_defined(yt)

        rows.append(
            {
                "drug": ev.drug,
                "n_test": len(yt_all),
                "pct_R": round(100 * float(yt_all.mean()), 1),
                # --- decision metrics, answered rows only ---
                "coverage": round(100 * float(mask.mean()), 1),
                "n_answered": int(mask.sum()),
                "bal_acc": round(_safe(balanced_accuracy_score, yt, yp), 3) if defined else nan,
                "recall_R": round(_safe(recall_score, yt, yp, pos_label=1, zero_division=0), 3) if defined else nan,
                "recall_S": round(_safe(recall_score, yt, yp, pos_label=0, zero_division=0), 3) if defined else nan,
                "f1_R": round(_safe(f1_score, yt, yp, pos_label=1, zero_division=0), 3) if defined else nan,
                # --- probabilistic metrics, all rows ---
                "auroc": round(_safe(roc_auc_score, yt_all, yp_all), 3),
                "pr_auc": round(_safe(average_precision_score, yt_all, yp_all), 3),
                "brier_raw": round(float(np.mean((ev.y_prob_raw - yt_all) ** 2)), 4),
                "brier_cal": round(float(np.mean((yp_all - yt_all) ** 2)), 4),
                "no_call": round(100 * float(np.mean([c == "no_call" for c in ev.calls])), 1),
                "not_appl": round(
                    100 * float(np.mean([c == "not_applicable" for c in ev.calls])), 1
                ),
            }
        )
    return pd.DataFrame(rows)


#: Metrics worth averaging across seeds. Counts (n_test) are reported separately.
_SEED_METRICS = (
    "coverage", "bal_acc", "recall_R", "recall_S", "f1_R",
    "auroc", "pr_auc", "brier_raw", "brier_cal", "no_call",
)


def multi_seed_metrics(
    dataset: Dataset, seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
) -> pd.DataFrame:
    """
    Per-drug metrics as mean ± sd over several grouped splits.

    A single split is not a result at this sample size. On the real E. coli data
    each drug's test slice is 9-11 genomes and only a fraction of those get
    answered, so seed 0 alone reported bal_acc = 1.000 for ampicillin off TWO
    answered rows. Re-drawing the split moves that number between 0.5 and 1.0.
    Reporting the mean and spread over seeds is the difference between a metric
    and an anecdote.

    `n_seeds` counts the seeds that actually produced a scoreable value for that
    metric — a seed whose test slice came back single-class, or where the model
    answered nothing, contributes nothing and is not silently averaged as zero.
    Read a low n_seeds as a warning about the metric, not a detail.
    """
    per_seed = []
    for seed in seeds:
        predictor = Predictor.fit(dataset, seed=seed)
        calibrator = Calibrator.fit(dataset, predictor)
        evaluations = [
            evaluate_drug(dataset, predictor, calibrator, drug)
            for drug in predictor.models
        ]
        table = metrics_table(evaluations)
        table["seed"] = seed
        per_seed.append(table)

    stacked = pd.concat(per_seed, ignore_index=True)

    rows = []
    for drug, group in stacked.groupby("drug", sort=False):
        row = {
            "drug": drug,
            "n_test": int(round(group["n_test"].mean())),
            "seeds": len(group),
        }
        for metric in _SEED_METRICS:
            values = group[metric].dropna()
            row[metric] = (
                f"{values.mean():.3f}±{values.std(ddof=0):.3f}" if len(values) else "n/a"
            )
            row[f"{metric}_n"] = len(values)
        rows.append(row)

    table = pd.DataFrame(rows)
    # Keep the per-metric seed counts out of the default view unless they differ
    # from the full set — then they matter and should be impossible to miss.
    count_cols = [f"{m}_n" for m in _SEED_METRICS]
    if (table[count_cols] == len(seeds)).all().all():
        table = table.drop(columns=count_cols)
    return table


def per_cluster_table(ev: DrugEvaluation, min_size: int = 3) -> pd.DataFrame:
    """
    Accuracy broken down by genetic cluster.

    Every test cluster is unseen in training by construction, so this is not
    "seen vs unseen" — it is whether performance is uniform across lineages or
    carried by one easy clade. A model that is excellent on two big clusters and
    coin-flip elsewhere is not one we can deploy.
    """
    mask = ev.answered
    rows = []
    for cluster in sorted(set(ev.clusters)):
        sel = (ev.clusters == cluster) & mask
        if sel.sum() < min_size:
            continue
        yt, yp = ev.y_true[sel], ev.y_pred[sel]
        # Clusters are often phenotypically homogeneous, so most contain a single
        # class. Balanced accuracy is undefined there — sklearn would quietly
        # return 0.5 or 1.0, which reads as "chance" or "perfect" when it means
        # neither. Report NaN and let plain accuracy carry those rows.
        single_class = len(np.unique(yt)) < 2
        rows.append(
            {
                "cluster": cluster,
                "n_answered": int(sel.sum()),
                "pct_R": round(100 * float(yt.mean()), 1),
                "accuracy": round(float((yt == yp).mean()), 3),
                "bal_acc": (
                    float("nan")
                    if single_class
                    else round(_safe(balanced_accuracy_score, yt, yp), 3)
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n_answered", ascending=False)


# --------------------------------------------------------------------------
# 2. The headline: random split vs grouped split
# --------------------------------------------------------------------------


def _random_split_like(split: GroupedSplit, n: int, seed: int) -> GroupedSplit:
    """A row-random split with the same slice sizes — the leaky baseline."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr, n_cal = len(split.train), len(split.calibration)
    return GroupedSplit(
        train=np.sort(perm[:n_tr]),
        calibration=np.sort(perm[n_tr : n_tr + n_cal]),
        test=np.sort(perm[n_tr + n_cal :]),
    )


def leakage_comparison(
    dataset: Dataset,
    drugs: list[str] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> pd.DataFrame:
    """
    Balanced accuracy under a random row split vs a grouped split, per drug.

    This is the pitch, and it must be reported honestly: average over several
    seeds, because a single-seed gap swings by +/-0.05. A positive gap means the
    random split was inflated by near-identical genomes spanning train and test.
    A gap near zero means that drug's signal did not depend on leakage — say so
    rather than quietly dropping the drug from the table.
    """
    from predictor import fit_drug_model

    records = []
    for drug in drugs or dataset.drugs:
        X, y, groups = dataset.xy_for_drug(drug)
        y = np.asarray(y)
        groups = np.asarray(groups)
        for seed in seeds:
            grouped = grouped_split(groups, y=y, seed=seed)
            random = _random_split_like(grouped, len(y), seed)

            for label, split in (("grouped", grouped), ("random", random)):
                try:
                    model = fit_drug_model(dataset, drug, split.train, groups)
                except Exception:
                    continue
                X_test, y_test = X.iloc[split.test], y[split.test]
                pred = np.array(
                    [
                        1 if model.probability_resistant(r) >= 0.5 else 0
                        for _, r in X_test.iterrows()
                    ]
                )
                records.append(
                    {
                        "drug": drug,
                        "split": label,
                        "seed": seed,
                        "bal_acc": _safe(balanced_accuracy_score, y_test, pred),
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    mean = df.pivot_table(index="drug", columns="split", values="bal_acc", aggfunc="mean")
    sd = df.pivot_table(index="drug", columns="split", values="bal_acc", aggfunc="std")
    out = pd.DataFrame(
        {
            "random": mean["random"].round(3),
            "random_sd": sd["random"].round(3),
            "grouped": mean["grouped"].round(3),
            "grouped_sd": sd["grouped"].round(3),
        }
    )
    out["gap"] = (out["random"] - out["grouped"]).round(3)
    return out.reset_index().sort_values("gap", ascending=False)


# --------------------------------------------------------------------------
# 3. Plots
# --------------------------------------------------------------------------


def reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int):
    """Bin centres and observed frequencies for a reliability diagram."""
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (y_prob >= lo) & ((y_prob < hi) if hi < 1.0 else (y_prob <= hi))
        if sel.sum() == 0:
            continue
        xs.append(float(y_prob[sel].mean()))
        ys.append(float(y_true[sel].mean()))
        ns.append(int(sel.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def plot_dashboard(
    evaluations: list[DrugEvaluation],
    table: pd.DataFrame,
    comparison: pd.DataFrame | None,
    out_path: Path,
    banner: str = BANNER,
) -> Path:
    """Four-panel summary figure: calibration, decisions, coverage, leakage."""
    import matplotlib

    matplotlib.use("Agg")  # headless — writes a file, never opens a window
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Genome Firewall — pipeline evaluation (held-out grouped test slice)",
        fontsize=14,
        fontweight="bold",
    )

    # -- (a) reliability diagram -------------------------------------------
    ax = axes[0][0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for ev in evaluations:
        xs, ys, _ = reliability_points(ev.y_true, ev.y_prob, N_RELIABILITY_BINS)
        if len(xs):
            ax.plot(xs, ys, "o-", ms=5, label=ev.drug, alpha=0.85)
    ax.axvspan(NO_CALL_LOW, NO_CALL_HIGH, color="grey", alpha=0.12)
    ax.text((NO_CALL_LOW + NO_CALL_HIGH) / 2, 0.03, "no-call band",
            ha="center", fontsize=8, color="dimgrey")
    ax.set_xlabel("calibrated P(resistant)")
    ax.set_ylabel("observed fraction resistant")
    ax.set_title("(a) Calibration — closer to the diagonal is better")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # -- (b) decision metrics ----------------------------------------------
    ax = axes[0][1]
    drugs = table["drug"].tolist()
    x = np.arange(len(drugs))
    width = 0.26
    for i, (col, lbl) in enumerate(
        [
            ("bal_acc", "balanced acc"),
            ("recall_R", "recall (resistant)"),
            ("recall_S", "recall (susceptible)"),
        ]
    ):
        ax.bar(x + (i - 1) * width, table[col].fillna(0), width, label=lbl)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.text(len(drugs) - 0.5, 0.51, "chance", fontsize=8, ha="right", color="dimgrey")
    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Decision quality — answered predictions only")
    ax.legend(fontsize=8)

    # -- (c) coverage vs abstention ----------------------------------------
    ax = axes[1][0]
    answered, no_call = table["coverage"], table["no_call"]
    not_appl = table["not_appl"]
    ax.bar(x, answered, 0.55, label="answered", color="#2a9d8f")
    ax.bar(x, no_call, 0.55, bottom=answered, label="no-call", color="#e9c46a")
    ax.bar(x, not_appl, 0.55, bottom=answered + no_call,
           label="not applicable (target absent)", color="#adb5bd")
    for xi, acc in enumerate(table["bal_acc"]):
        if not np.isnan(acc):
            ax.text(xi, 4, f"{acc:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% of test genomes")
    ax.set_title("(c) Coverage — bold = balanced acc bought by that coverage")
    ax.legend(fontsize=8)

    # -- (d) leakage comparison --------------------------------------------
    ax = axes[1][1]
    if comparison is not None and not comparison.empty:
        cd = comparison["drug"].tolist()
        cx = np.arange(len(cd))
        ax.bar(cx - 0.2, comparison["random"], 0.4, yerr=comparison["random_sd"],
               capsize=3, label="random split (leaky)", color="#e76f51")
        ax.bar(cx + 0.2, comparison["grouped"], 0.4, yerr=comparison["grouped_sd"],
               capsize=3, label="grouped split (honest)", color="#264653")
        for xi, gap in enumerate(comparison["gap"]):
            ax.text(xi, 1.01, f"{gap:+.3f}", ha="center", fontsize=9,
                    fontweight="bold" if abs(gap) >= 0.05 else "normal",
                    color="#e76f51" if gap >= 0.05 else "dimgrey")
        ax.set_xticks(cx)
        ax.set_xticklabels(cd, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_ylabel("balanced accuracy")
        ax.set_title("(d) Why grouped splitting matters (mean ± sd over seeds)")
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.axis("off")

    fig.text(0.5, 0.005, banner, ha="center", fontsize=10,
             color="#b00020", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------


def run_full_evaluation(
    data_dir: Path,
    out_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> None:
    """End-to-end: load -> fit -> calibrate -> evaluate -> plot."""
    from data_io import load_dataset, summarize

    print("=" * 78)
    print("Genome Firewall — pipeline evaluation")
    print("=" * 78)
    banner = banner_for(data_dir)
    print(f"\n!! {banner}\n")

    ds = load_dataset(data_dir)
    print(f"{len(ds.features)} genomes, {len(ds.feature_names)} features, "
          f"{ds.genomes['cluster_id'].nunique()} clusters")
    print("\nDataset")
    print(summarize(ds).to_string(index=False))

    predictor = Predictor.fit(ds, seed=0)
    calibrator = Calibrator.fit(ds, predictor)
    evaluations = [
        evaluate_drug(ds, predictor, calibrator, drug) for drug in predictor.models
    ]

    table = metrics_table(evaluations)

    # The headline table. Reported before the seed-0 detail deliberately: the
    # single-split numbers below are one draw from the distribution summarised
    # here, and quoting them on their own overstates what this data supports.
    seed_table = multi_seed_metrics(ds, seeds=seeds)
    print(f"\nPer-drug metrics — mean±sd over {len(seeds)} grouped splits  [REPORT THESE]")
    print(seed_table.to_string(index=False))

    print("\nSeed 0 detail (one draw — for the dashboard plots, not for quoting)")
    print(table.to_string(index=False))

    print("\nPer-cluster breakdown — is performance uniform across lineages?")
    for ev in evaluations:
        per_cluster = per_cluster_table(ev)
        if per_cluster.empty:
            print(f"\n  {ev.drug}: no cluster has enough answered rows to break down")
            continue
        spread = per_cluster["accuracy"].max() - per_cluster["accuracy"].min()
        print(f"\n  {ev.drug}  (accuracy spread across clusters: {spread:.3f})")
        print(per_cluster.head(6).to_string(index=False))

    print(f"\nRandom vs grouped split, mean over {len(seeds)} seeds")
    comparison = leakage_comparison(ds, seeds=seeds)
    print(comparison.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dashboard(evaluations, table, comparison, out_dir / "evaluation.png", banner)
    print(f"\nDashboard written to {path}")

    table.to_csv(out_dir / "metrics_seed0.csv", index=False)
    seed_table.to_csv(out_dir / "metrics.csv", index=False)
    comparison.to_csv(out_dir / "leakage_comparison.csv", index=False)
    print(f"Tables written to {out_dir / 'metrics.csv'} (multi-seed), "
          f"{out_dir / 'metrics_seed0.csv'} and {out_dir / 'leakage_comparison.csv'}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    run_full_evaluation(root / "data" / "synthetic", root / "reports")
