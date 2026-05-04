##########################################################################
#                                                                        #
#  compare_cv_models.py                                                  #
#  Author: Tommy Lubomirski                                              #
#  Created: Nov. 30/25                                                   #
#                                                                        #
#  Compare two CV runs from the saved ROOT outputs.                      #
#                                                                        #
##########################################################################

import os
import time
import logging
import argparse
import pandas as pd

# load CV fold ROOT outputs into per-fold pandas DataFrames
from .cv_structure import load_all_folds_as_dfs

# metrics + aggregation helpers (per-fold -> per-variable / per-parton summaries)
from .metrics import (
    compute_per_fold_metrics,
    aggregate_metrics_by_variable,
    aggregate_metrics_by_parton,
)

logger = logging.getLogger(__name__)

# these are the summary columns we care about for model vs model comparisons.
COMPARE_METRICS = [
    "mae_mean",
    "mse_mean",
    "err_mae_mean",
    "err_rmse_mean",
    "err_sigma68_mean",
]


def compute_cv_metrics_for_dir(cv_dir, base_vars=None):
    # normalize path so outputs/logging are consistent no matter how you call it
    cv_dir = os.path.abspath(cv_dir)
    logger.info("Computing CV metrics for %s", cv_dir)

    # load all folds
    start_time = time.time()
    fold_to_df = load_all_folds_as_dfs(
        cv_dir=cv_dir,
        results_subdir="results",
        name_substring=None,
        base_vars=base_vars,          # optionally restrict to a smaller variable set
        include_event_number=False,   # not needed for comparisons, keeps memory down
    )
    print(f"Loaded {len(fold_to_df)} folds in {time.time() - start_time:.2f} seconds.")

    # per-fold, per-variable metrics
    # This is the "raw" CV data we aggregate later.
    per_fold_df = compute_per_fold_metrics(
        fold_to_df=fold_to_df,
        base_vars=base_vars,
        require_both_truth_reco=True,  # if a var is missing, fail loud
    )

    # Aggregate summaries 
    by_var_df = aggregate_metrics_by_variable(per_fold_df)
    by_parton_df = aggregate_metrics_by_parton(per_fold_df)

    logger.info(
        "Finished metrics for %s: %d per-fold rows, %d variables, %d partons",
        cv_dir,
        len(per_fold_df),
        len(by_var_df),
        len(by_parton_df),
    )

    # keep everything so downstream scripts can choose what they need
    return {
        "per_fold": per_fold_df,
        "by_variable": by_var_df,
        "by_parton": by_parton_df,
    }


def _merge_metric_deltas(df_a, df_b, merge_keys, label_a, label_b):
    # inner merge so we only compare rows that exist in both runs
    merged = df_a.merge(
        df_b,
        on=list(merge_keys),
        how="inner",
        suffixes=(f"_{label_a}", f"_{label_b}"),
    )

    # avoid divide-by-zero if a metric is ~0 for some weird variable
    eps = 1e-12

    # compute absolute and relative deltas for each metric we care about
    for metric in COMPARE_METRICS:
        col_a = f"{metric}_{label_a}"
        col_b = f"{metric}_{label_b}"
        if col_a not in merged.columns or col_b not in merged.columns:
            # some tables might not have every metric
            continue

        merged[f"delta_{metric}"] = merged[col_b] - merged[col_a]
        merged[f"rel_delta_{metric}"] = (merged[col_b] - merged[col_a]) / (merged[col_a] + eps)

    return merged


def compare_by_variable(metrics_a, metrics_b, label_a="A", label_b="B"):
    # copy so we dont mutate the original dicts
    df_a = metrics_a["by_variable"].copy()
    df_b = metrics_b["by_variable"].copy()

    # variable-level tables are keyed by variable + its parsed tags
    merge_keys = ["variable", "parton", "observable"]

    # of error_mode exists, include it so we dont mix residual vs resolution rows
    if "error_mode" in df_a.columns and "error_mode" in df_b.columns:
        merge_keys.append("error_mode")

    merged = _merge_metric_deltas(df_a, df_b, merge_keys, label_a, label_b)

    # sort in a way thats easy to scan by eye
    return merged.sort_values(["parton", "observable", "variable"]).reset_index(drop=True)


def compare_by_parton(metrics_a, metrics_b, label_a="A", label_b="B"):
    df_a = metrics_a["by_parton"].copy()
    df_b = metrics_b["by_parton"].copy()

    merge_keys = ["parton"]

    # ff we computed parton summaries for specific observables (like pt-only),
    # keep that as part of the identity.
    if "observables" in df_a.columns and "observables" in df_b.columns:
        merge_keys.append("observables")

    merged = _merge_metric_deltas(df_a, df_b, merge_keys, label_a, label_b)
    return merged.sort_values("parton").reset_index(drop=True)


def _short_label_from_dir(cv_dir):
    # grab last directory name and strip _CV to get a decent default label
    parts = os.path.abspath(cv_dir).rstrip(os.sep).split(os.sep)
    if parts:
        return parts[-1].replace("_CV", "")
    return "model"


def main():
    parser = argparse.ArgumentParser(description="Compare two TRecNet CV runs.")
    parser.add_argument("cv_dir_a", help="First *_CV directory.")
    parser.add_argument("cv_dir_b", help="Second *_CV directory.")
    parser.add_argument(
        "--label-a", default=None, help="Short label for first model."
    )
    parser.add_argument(
        "--label-b", default=None, help="Short label for second model."
    )
    parser.add_argument(
        "--base-vars",
        nargs="+",
        default=None,
        help="Only use these base vars (for example: th_pt b1_pt).",
    )
    parser.add_argument(
        "--outdir",
        default="./cv_comparisons",
        help="Where to write the comparison CSVs.",
    )

    args = parser.parse_args()

    # keep logging readable on the command line
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # default labels from directory names if not explicitely given
    label_a = args.label_a or _short_label_from_dir(args.cv_dir_a)
    label_b = args.label_b or _short_label_from_dir(args.cv_dir_b)

    logger.info("Label A: %s  (dir: %s)", label_a, args.cv_dir_a)
    logger.info("Label B: %s  (dir: %s)", label_b, args.cv_dir_b)

    # compute tables fro each model
    metrics_a = compute_cv_metrics_for_dir(args.cv_dir_a, base_vars=args.base_vars)
    metrics_b = compute_cv_metrics_for_dir(args.cv_dir_b, base_vars=args.base_vars)

    # merge plus compute deltas
    var_cmp = compare_by_variable(metrics_a, metrics_b, label_a, label_b)
    parton_cmp = compare_by_parton(metrics_a, metrics_b, label_a, label_b)

    # output folder
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    var_csv = os.path.join(outdir, f"variable_comparison_{label_a}_vs_{label_b}.csv")
    parton_csv = os.path.join(outdir, f"parton_comparison_{label_a}_vs_{label_b}.csv")

    # save the actual comparison resulkts 
    var_cmp.to_csv(var_csv, index=False)
    parton_cmp.to_csv(parton_csv, index=False)

    #  cli summary so we can sanity-check without opening the CSV
    print("\n=== Saved comparison CSVs ===")
    print(f"Variable-level CSV: {var_csv}")
    print(f"Parton-level CSV:   {parton_csv}")

    print("\n=== Parton-level comparison ===")
    print(parton_cmp.to_string(index=False))

    print("\n=== Variable-level comparison (head) ===")
    print(var_cmp.head(30).to_string(index=False))


if __name__ == "__main__":
    main()