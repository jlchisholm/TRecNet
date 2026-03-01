##########################################################################
#                                                                        #
#  thesis_cv_report.py                                                   #
#  Author: Tommy Lubomirski                                              #
#  Updated: Mar. 1/26                                                    #
#                                                                        #
#  Compare two CV runs and dump out the extra plots + metrics.           #
#                                                                        #
##########################################################################

import os
import json
import logging
import argparse
from datetime import datetime

import pandas as pd

# CV i/o helpers
from .cv_structure import get_fold_to_file, load_all_folds_as_dfs, pooled_truth_reco

# metrics + aggregation
from .metrics import (
    aggregate_metrics_by_parton,
    aggregate_metrics_by_variable,
    aggregate_metrics_global,
    build_error_array,
    compute_error_metrics,
    compute_per_fold_metrics,
)

# thesis-style plots (truth-reco, residual overlays, etc)
from .thesis_plotting import (
    plot_metric_bar,
    plot_qq,
    plot_residual_overlay,
    plot_truth_reco_density,
)

# plot specs / axis ranges / defaults
from .thesis_specs import freeze_mass_range, get_variable_spec, load_thesis_defaults

logger = logging.getLogger(__name__)

# these should match columns produced by the aggregated tables
COMPARE_METRICS = [
    "mae_mean",
    "mse_mean",
    "err_mae_mean",
    "err_rmse_mean",
    "err_sigma68_mean",
]


def _trecnet_root():
    # repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _default_outdir():
    # default landing spot for comparison bundles
    return os.path.join(_trecnet_root(), "cv_comparisons")


def _short_label_from_dir(cv_dir):
    # use the directory name as the label unless user overrides it
    parts = os.path.abspath(cv_dir).rstrip(os.sep).split(os.sep)
    if parts:
        return parts[-1].replace("_CV", "")
    return "model"


def _ensure_dir(path):
    # mkdir -p
    os.makedirs(path, exist_ok=True)
    return path


def _write_csv(df, outpath):
    # little helper so I dont repeat mkdirs everywhere
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)


def _compare_tables(df_a, df_b, merge_keys, label_a, label_b):
    # merge two summary tables and compute deltas (abs + rel)
    merged = df_a.merge(
        df_b,
        on=list(merge_keys),
        how="inner",
        suffixes=(f"_{label_a}", f"_{label_b}"),
    )

    eps = 1e-12  # safe divide
    for metric in COMPARE_METRICS:
        col_a = f"{metric}_{label_a}"
        col_b = f"{metric}_{label_b}"
        if col_a not in merged.columns or col_b not in merged.columns:
            # not every table has every metric depending on config
            continue
        merged[f"delta_{metric}"] = merged[col_b] - merged[col_a]
        merged[f"rel_delta_{metric}"] = (merged[col_b] - merged[col_a]) / (merged[col_a] + eps)

    return merged


def _validated_fold_map(cv_dir, expected_folds):
    # make sure we actually found the folds we think we found
    fold_to_file = get_fold_to_file(cv_dir=cv_dir)
    if len(fold_to_file) != expected_folds:
        raise RuntimeError(
            f"Expected {expected_folds} folds in {cv_dir}, found {len(fold_to_file)}."
        )
    return fold_to_file


def _compute_model_tables(fold_to_df, key_vars):
    # compute the per-fold long table, then the summaries we want to save/compare
    per_fold = compute_per_fold_metrics(fold_to_df)
    by_variable = aggregate_metrics_by_variable(per_fold)

    # pt-only per-parton summary (mixing pt/eta/phi/m in one number is kinda cursed)
    by_parton_pt = aggregate_metrics_by_parton(per_fold, observables=["pt"])

    # global summary over the key vars 
    global_keyvars = aggregate_metrics_global(per_fold, variables=key_vars)

    return {
        "per_fold": per_fold,
        "by_variable": by_variable,
        "by_parton_pt": by_parton_pt,
        "global_keyvars": global_keyvars,
    }


def _save_model_tables(tables, label, outdir):
    # save the tables for one model into outdir/metrics/
    metrics_dir = _ensure_dir(os.path.join(outdir, "metrics"))
    _write_csv(tables["per_fold"], os.path.join(metrics_dir, f"per_fold_{label}.csv"))
    _write_csv(tables["by_variable"], os.path.join(metrics_dir, f"by_variable_{label}.csv"))
    _write_csv(tables["by_parton_pt"], os.path.join(metrics_dir, f"by_parton_pt_{label}.csv"))
    _write_csv(tables["global_keyvars"], os.path.join(metrics_dir, f"global_keyvars_{label}.csv"))


def _plot_key_variables(
    key_vars,
    fold_to_df_a,
    fold_to_df_b,
    label_a,
    label_b,
    color_a,
    color_b,
    defaults,
    fig_format,
    outdir,
):
    # make the main plots for a short list of key variables
    atlas_label = defaults["atlas_label"]
    range_manifest = {}

    for base_var in key_vars:
        # pull plot settings (axis labels, binning, ranges, etc)
        var_spec = get_variable_spec(base_var, defaults)

        # pool across folds for nicer-looking plots (tables still use fold stats)
        truth_a, reco_a = pooled_truth_reco(fold_to_df_a, base_var)
        truth_b, reco_b = pooled_truth_reco(fold_to_df_b, base_var)

        # convert to "error space" for residual plots (pt resolution, wrapped phi, etc)
        _, _, err_a = build_error_array(truth_a, reco_a, base_var)
        _, _, err_b = build_error_array(truth_b, reco_b, base_var)

        # if residual range isnt defined (usually for masses), freeze it from model A once
        # so A and B always plot on the same axis
        if var_spec["residual_range"] is None:
            var_spec["residual_range"] = freeze_mass_range(err_a)

        # keep a record of ranges in the manifest so plots are reproducible
        range_manifest[base_var] = {
            "truth_range": list(var_spec["truth_range"]),
            "residual_range": list(var_spec["residual_range"]),
        }

        # truth vs reco plots (one per model)
        plot_truth_reco_density(
            truth_a,
            reco_a,
            var_spec,
            label_a,
            os.path.join(outdir, "figs", "truth_reco", label_a, f"{base_var}.{fig_format}"),
            atlas_label=atlas_label,
        )
        plot_truth_reco_density(
            truth_b,
            reco_b,
            var_spec,
            label_b,
            os.path.join(outdir, "figs", "truth_reco", label_b, f"{base_var}.{fig_format}"),
            atlas_label=atlas_label,
        )

        # overlay residuals 
        plot_residual_overlay(
            err_a,
            err_b,
            compute_error_metrics(err_a),
            compute_error_metrics(err_b),
            var_spec,
            label_a,
            label_b,
            color_a,
            color_b,
            os.path.join(outdir, "figs", "residual_overlay", f"{base_var}.{fig_format}"),
            atlas_label=atlas_label,
        )

    return range_manifest


def _plot_qq_variables(qq_vars, fold_to_df_a, defaults, label_a, fig_format, outdir):
    #  Q-Q plots  to check tails
    for base_var in qq_vars:
        var_spec = get_variable_spec(base_var, defaults)

        truth_a, reco_a = pooled_truth_reco(fold_to_df_a, base_var)
        _, _, err_a = build_error_array(truth_a, reco_a, base_var)

        # if no residual range is defined, pick one from the data
        if var_spec["residual_range"] is None:
            var_spec["residual_range"] = freeze_mass_range(err_a)

        plot_qq(
            err_a,
            var_spec,
            label_a,
            os.path.join(outdir, "figs", "qq", f"{base_var}.{fig_format}"),
            atlas_label=defaults["atlas_label"],
        )


def main():
    parser = argparse.ArgumentParser(description="Compare two *_CV directories.")
    parser.add_argument("cv_dir_a", help="First *_CV directory.")
    parser.add_argument("cv_dir_b", help="Second *_CV directory.")
    parser.add_argument("--label-a", default=None, help="Short label for model A.")
    parser.add_argument("--label-b", default=None, help="Short label for model B.")
    parser.add_argument("--outdir", default=None, help="Where to write everything.")
    parser.add_argument(
        "--key-vars",
        nargs="+",
        default=None,
        help="Key vars for the plots and global summary.",
    )
    parser.add_argument("--color-a", default=None, help="Color for model A.")
    parser.add_argument("--color-b", default=None, help="Color for model B.")
    parser.add_argument("--expected-folds", type=int, default=8, help="How many folds to expect.")
    parser.add_argument("--fig-format", default="pdf", help="Figure format.")
    parser.add_argument("--make-qq", action="store_true", help="Also make Q-Q plots.")
    parser.add_argument(
        "--qq-vars",
        nargs="+",
        default=None,
        help="Vars for Q-Q plots.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # load plotting defaults from json configs (and some hardcoded fallbacks)
    defaults = load_thesis_defaults()

    # labels plus plot list defaults
    label_a = args.label_a or _short_label_from_dir(args.cv_dir_a)
    label_b = args.label_b or _short_label_from_dir(args.cv_dir_b)
    key_vars = args.key_vars or defaults["key_vars_default"]
    qq_vars = args.qq_vars or defaults["qq_vars_default"]
    color_a = args.color_a or defaults["color_a"]
    color_b = args.color_b or defaults["color_b"]

    # default output folder is deterministic so I dont lose runs
    if args.outdir is None:
        outdir = os.path.join(_default_outdir(), f"{label_a}_vs_{label_b}")
    else:
        outdir = os.path.abspath(args.outdir)

    # fail early if a CV directory is missing folds
    fold_to_file_a = _validated_fold_map(args.cv_dir_a, args.expected_folds)
    fold_to_file_b = _validated_fold_map(args.cv_dir_b, args.expected_folds)

    # load fold DataFrames (truth_* and reco_* columns)
    fold_to_df_a = load_all_folds_as_dfs(
        args.cv_dir_a,
        include_event_number=False,
    )
    fold_to_df_b = load_all_folds_as_dfs(
        args.cv_dir_b,
        include_event_number=False,
    )

    # compute and save metrics tables for each model
    tables_a = _compute_model_tables(fold_to_df_a, key_vars)
    tables_b = _compute_model_tables(fold_to_df_b, key_vars)

    _save_model_tables(tables_a, label_a, outdir)
    _save_model_tables(tables_b, label_b, outdir)

    # compare the summary tables and compute delta/rel_delta columns
    compare_by_variable = _compare_tables(
        tables_a["by_variable"],
        tables_b["by_variable"],
        ["variable", "parton", "observable", "error_mode"],
        label_a,
        label_b,
    )
    compare_by_parton_pt = _compare_tables(
        tables_a["by_parton_pt"],
        tables_b["by_parton_pt"],
        ["parton", "observables"],
        label_a,
        label_b,
    )
    compare_global_keyvars = _compare_tables(
        tables_a["global_keyvars"],
        tables_b["global_keyvars"],
        ["summary", "variables"],
        label_a,
        label_b,
    )

    metrics_dir = _ensure_dir(os.path.join(outdir, "metrics"))
    _write_csv(compare_by_variable, os.path.join(metrics_dir, "compare_by_variable.csv"))
    _write_csv(compare_by_parton_pt, os.path.join(metrics_dir, "compare_by_parton_pt.csv"))
    _write_csv(compare_global_keyvars, os.path.join(metrics_dir, "compare_global_keyvars.csv"))

    # make the main plot bundle (truth-reco + residual overlays)
    range_manifest = _plot_key_variables(
        key_vars,
        fold_to_df_a,
        fold_to_df_b,
        label_a,
        label_b,
        color_a,
        color_b,
        defaults,
        args.fig_format,
        outdir,
    )

    # pull out the key vars in a fixed order for the summary bar plot
    keyvar_rows = compare_by_variable[compare_by_variable["variable"].isin(key_vars)].copy()
    keyvar_rows["variable"] = pd.Categorical(
        keyvar_rows["variable"],
        categories=list(key_vars),
        ordered=True,
    )
    keyvar_rows = keyvar_rows.sort_values("variable").reset_index(drop=True)
    keyvar_rows["variable"] = keyvar_rows["variable"].astype(str)

    #  sigma68 per key variable, with error bars
    plot_metric_bar(
        keyvar_rows,
        label_a,
        label_b,
        "err_sigma68_mean",
        "variable",
        os.path.join(outdir, "figs", "summary", f"keyvars_err_sigma68.{args.fig_format}"),
        ylabel=r"$\sigma_{68}$",
        title="Key-variable resolution summary",
        color_a=color_a,
        color_b=color_b,
        atlas_label=defaults["atlas_label"],
    )

    # Q-Q plots (currently only for model A, mostly as a diagnostic)
    if args.make_qq:
        _plot_qq_variables(
            qq_vars,
            fold_to_df_a,
            defaults,
            label_a,
            args.fig_format,
            outdir,
        )

    # save a history so I can reproduce plots later without guessing config/ranges
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "cv_dir_a": os.path.abspath(args.cv_dir_a),
            "cv_dir_b": os.path.abspath(args.cv_dir_b),
            "label_a": label_a,
            "label_b": label_b,
        },
        "fold_root_files": {
            label_a: fold_to_file_a,
            label_b: fold_to_file_b,
        },
        "key_vars": list(key_vars),
        "qq_vars": list(qq_vars) if args.make_qq else [],
        "figure_format": args.fig_format,
        "plot_ranges": range_manifest,
        "atlas_label": defaults["atlas_label"],
        "metric_definitions": {
            "legacy": "mae, mse, rel_mae, rel_mse come from reco - truth after finite masking.",
            "style_aligned": "err_* uses pt resolution, wrapped phi residuals, and plain residuals otherwise.",
            "sigma68": "0.5 * (q84 - q16)",
            "tail_frac_3sigma": "mean(abs(err) > 3 * sigma68)",
        },
    }

    with open(os.path.join(outdir, "manifest.json"), "w") as outfile:
        json.dump(manifest, outfile, indent=2)

    logger.info("Wrote comparison outputs to %s", outdir)


if __name__ == "__main__":
    main()