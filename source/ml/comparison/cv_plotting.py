##########################################################################
#                                                                        #
#  cv_plotting.py                                                        #
#  Author: Tommy Lubomirski.                                             #
#  Created: Nov. 30/25                                                   #
#                                                                        #
#  Quick plots for CV comparison csvs.                                   #
#                                                                        #
##########################################################################

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless saving: no gui
import matplotlib.pyplot as plt

# rough grouping so plots arent just a wall of gray bars
HADRONIC_PREFIXES = ("th", "wh", "ttbar", "b1", "b2")
LEPTONIC_PREFIXES = ("tl", "wl", "lep")


def _ensure_dir(path):
    # make sure parent directory exists
    path = os.path.abspath(path)
    outdir = os.path.dirname(path)
    os.makedirs(outdir, exist_ok=True)
    return path


def branch_of(parton_or_var):
    # decide whether something is hadronic/leptonic based on the prefix
    name = parton_or_var
    for p in HADRONIC_PREFIXES:
        if name.startswith(p + "_") or name == p:
            return "had"
    for p in LEPTONIC_PREFIXES:
        if name.startswith(p + "_") or name == p:
            return "lep"
    return "other"


def color_for(parton_or_var):
    # keep colors consistent across plots
    br = branch_of(parton_or_var)
    if br == "had":
        return "royalblue"
    if br == "lep":
        return "darkorange"
    return "0.6"


def hatch_for(delta):
    # quick visual cue for sign
    # positive means worse if metric is "smaller is better"
    if delta > 0:
        return "///"
    if delta < 0:
        return "\\\\\\"
    return ""


def plot_parton_mae_bar(
    parton_cmp,
    label_a,
    label_b,
    outpath,
    width=0.35,
):
    outpath = _ensure_dir(outpath)

    # x-axis = partons, y-axis = mean MAE per parton
    partons = parton_cmp["parton"].tolist()
    x = np.arange(len(partons))

    mae_a = parton_cmp[f"mae_mean_{label_a}"].to_numpy()
    mae_b = parton_cmp[f"mae_mean_{label_b}"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, mae_a, width=width, label=label_a, alpha=0.85)
    plt.bar(x + width / 2, mae_b, width=width, label=label_b, alpha=0.85)

    plt.xticks(x, partons)
    plt.ylabel("MAE (mean over variables & folds)")
    plt.title(f"Per-parton MAE comparison: {label_a} vs {label_b}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[cv_plotting] Saved parton MAE bar plot to: {outpath}")


def plot_parton_delta_mae(parton_cmp, label_a, label_b, outpath):
    outpath = _ensure_dir(outpath)

    # prefer relative delta if it's there, otherwise fall back to whatever we can compute
    if "rel_delta_mae" in parton_cmp.columns:
        metric_col = "rel_delta_mae"
        ylab = r"relative $\Delta$MAE = (MAE_b − MAE_a) / MAE_a"
    elif "delta_rel_mae_mean" in parton_cmp.columns:
        metric_col = "delta_rel_mae_mean"
        ylab = r"$\Delta$relMAE = relMAE_b − relMAE_a"
    else:
        metric_col = "delta_mae_mean"
        ylab = r"$\Delta$MAE = MAE_b − MAE_a"

    partons = parton_cmp["parton"].tolist()
    x = np.arange(len(partons))
    delta = parton_cmp[metric_col].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.axhline(0.0, color="black", linewidth=1)  # "no change" line
    bars = plt.bar(
        x,
        delta,
        alpha=0.85,
        color=[color_for(p) for p in partons],
    )
    for bar, d in zip(bars, delta):
        bar.set_hatch(hatch_for(d))

    plt.xticks(x, partons)
    plt.ylabel(ylab)
    plt.title(f"Per-parton {metric_col}: {label_b} vs {label_a}")
    plt.tight_layout()

    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[cv_plotting] Saved parton $\\Delta$MAE plot to: {outpath}")


def _delta_xlabel(metric, label_a, label_b):
    # make the axis label readable without having to remember the column naming
    if metric.startswith("rel_delta_"):
        base = metric.replace("rel_delta_", "").upper()
        return f"rel Δ{base} = ({base}({label_b}) − {base}({label_a})) / {base}({label_a})"
    if metric.startswith("delta_"):
        base = metric.replace("delta_", "").upper()
        return f"Δ{base} = {base}({label_b}) − {base}({label_a})"
    return f"{metric}  ({label_b} − {label_a})"


def winbar_variable_delta(
    var_cmp,
    label_a,
    label_b,
    metric="delta_mse_mean",
    title_prefix="",
    outpath=None,
):
    # big horizontal bar plot of deltas for every variable
    if metric not in var_cmp.columns:
        raise ValueError(f"metric '{metric}' not found in var_cmp columns.")

    d = var_cmp.copy()
    d["delta"] = d[metric]
    d["branch"] = d["variable"].map(branch_of)
    d.sort_values("delta", inplace=True)  # so "best improvements" are on top

    # simple win/loss count (negative delta = B better if smaller is better)
    n_b_better = (d["delta"] < 0).sum()
    n_a_better = (d["delta"] > 0).sum()
    n_tie = (d["delta"] == 0).sum()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(d))))
    bars = ax.barh(
        d["variable"],
        d["delta"],
        color=[color_for(v) for v in d["variable"]],
        alpha=0.9,
    )
    for bar, delta in zip(bars, d["delta"]):
        bar.set_hatch(hatch_for(delta))

    ax.axvline(0, linewidth=1.5, color="black", alpha=0.6)
    ax.set_xlabel(_delta_xlabel(metric, label_a, label_b))
    ax.set_ylabel("Variable")

    if not title_prefix:
        title_prefix = metric

    ax.set_title(
        f"{title_prefix} by variable\n"
        f"(blue = hadronic, orange = leptonic) | "
        f"{label_b} better: {n_b_better}, {label_a} better: {n_a_better}, tie: {n_tie}"
    )
    plt.tight_layout()

    if outpath is not None:
        outpath = _ensure_dir(outpath)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"[cv_plotting] Saved variable winbar plot to: {outpath}")
        plt.close()
    else:
        plt.show()


def plot_topk_variable_delta(
    var_cmp,
    label_a,
    label_b,
    metric="delta_mse_mean",
    top_k=10,
    ascending=True,
    outpath=None,
    include_parton_in_label=True,
):
    # smaller "top-k" plot
    if metric not in var_cmp.columns:
        raise ValueError(f"metric '{metric}' not found in var_cmp columns.")

    df_sorted = var_cmp.sort_values(metric, ascending=ascending).copy()
    df_top = df_sorted.head(top_k)

    if df_top.empty:
        print("[cv_plotting] No variables to plot for top-k; skipping.")
        return

    # preappend parton just so its obvious what we're looking at
    if include_parton_in_label:
        labels = (df_top["parton"] + ":" + df_top["variable"]).tolist()
    else:
        labels = df_top["variable"].tolist()

    x = np.arange(len(df_top))
    delta_vals = df_top[metric].to_numpy()

    plt.figure(figsize=(10, max(4, 0.3 * top_k)))
    plt.axhline(0.0, color="black", linewidth=1)
    bars = plt.barh(
        x,
        delta_vals,
        alpha=0.85,
        color=[color_for(v) for v in df_top["variable"]],
    )
    for bar, dval in zip(bars, delta_vals):
        bar.set_hatch(hatch_for(dval))

    # symmetric x-limits so +/- are easy to compare
    lim = np.nanmax(np.abs(delta_vals)) * 1.05
    plt.xlim(-lim, lim)

    plt.yticks(x, labels)
    plt.xlabel(_delta_xlabel(metric, label_a, label_b))
    direction = "best improvements" if ascending else "largest regressions"
    plt.title(f"Top-{top_k} variables by {metric} ({direction})")
    plt.tight_layout()

    if outpath is not None:
        outpath = _ensure_dir(outpath)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"[cv_plotting] Saved top-k variable $\\Delta$ plot to: {outpath}")
        plt.close()
    else:
        plt.show()


def plot_parton_metric_bar(
    parton_cmp,
    label_a,
    label_b,
    metric_base,
    outpath,
    width=0.35,
    ylabel=None,
    title=None,
):
    # generic per-parton side-by-side bar plot (works for rel_mae_mean, rel_mse_mean, etc)
    outpath = _ensure_dir(outpath)

    col_a = f"{metric_base}_{label_a}"
    col_b = f"{metric_base}_{label_b}"
    if col_a not in parton_cmp.columns or col_b not in parton_cmp.columns:
        print(f"[cv_plotting] Missing {col_a} or {col_b}; skipping {metric_base} bar plot")
        return

    partons = parton_cmp["parton"].tolist()
    x = np.arange(len(partons))
    a = parton_cmp[col_a].to_numpy()
    b = parton_cmp[col_b].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, a, width=width, label=label_a, alpha=0.85)
    plt.bar(x + width / 2, b, width=width, label=label_b, alpha=0.85)
    plt.xticks(x, partons)

    plt.ylabel(ylabel or metric_base)
    plt.title(title or f"Per-parton {metric_base}: {label_a} vs {label_b}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[cv_plotting] Saved {metric_base} bar plot to: {outpath}")


def plot_from_csvs(parton_csv, variable_csv, label_a, label_b, outdir):
    # load the comparison csvs from compare_cv_models.py
    parton_cmp = pd.read_csv(parton_csv)
    var_cmp = pd.read_csv(variable_csv)

    # some deltas can be derived if the comparison script didnt already include them
    for base in ["rel_mae_mean", "rel_mse_mean"]:
        a = f"{base}_{label_a}"
        b = f"{base}_{label_b}"
        if a in parton_cmp.columns and b in parton_cmp.columns:
            parton_cmp[f"delta_{base}"] = parton_cmp[b] - parton_cmp[a]
        if a in var_cmp.columns and b in var_cmp.columns:
            var_cmp[f"delta_{base}"] = var_cmp[b] - var_cmp[a]

    # default output folder = same folder as the csvs
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(parton_csv))
    eps = 1e-12  # safe divide

    # column names depend on model labels
    mae_a_p = f"mae_mean_{label_a}"
    mae_b_p = f"mae_mean_{label_b}"
    mse_a_p = f"mse_mean_{label_a}"
    mse_b_p = f"mse_mean_{label_b}"

    mae_a_v = f"mae_mean_{label_a}"
    mae_b_v = f"mae_mean_{label_b}"
    mse_a_v = f"mse_mean_{label_a}"
    mse_b_v = f"mse_mean_{label_b}"

    # relative delta is nice because its dimensionless (and comparable across stuff)
    if mse_a_p in parton_cmp.columns and mse_b_p in parton_cmp.columns:
        parton_cmp["rel_delta_mse"] = (
            parton_cmp[mse_b_p] - parton_cmp[mse_a_p]
        ) / (parton_cmp[mse_a_p] + eps)
    if mae_a_p in parton_cmp.columns and mae_b_p in parton_cmp.columns:
        parton_cmp["rel_delta_mae"] = (
            parton_cmp[mae_b_p] - parton_cmp[mae_a_p]
        ) / (parton_cmp[mae_a_p] + eps)

    if mse_a_v in var_cmp.columns and mse_b_v in var_cmp.columns:
        var_cmp["rel_delta_mse"] = (
            var_cmp[mse_b_v] - var_cmp[mse_a_v]
        ) / (var_cmp[mse_a_v] + eps)
    if mae_a_v in var_cmp.columns and mae_b_v in var_cmp.columns:
        var_cmp["rel_delta_mae"] = (
            var_cmp[mae_b_v] - var_cmp[mae_a_v]
        ) / (var_cmp[mae_a_v] + eps)

    # legacy compatibility: some older scripts expect this name
    col_a = f"rel_mae_mean_{label_a}"
    col_b = f"rel_mae_mean_{label_b}"
    if col_a in parton_cmp.columns and col_b in parton_cmp.columns:
        parton_cmp["delta_rel_mae_mean"] = parton_cmp[col_b] - parton_cmp[col_a]

    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    # make plots 

    plot_parton_mae_bar(
        parton_cmp,
        label_a=label_a,
        label_b=label_b,
        outpath=os.path.join(outdir, f"parton_mae_{label_a}_vs_{label_b}.png"),
    )

    plot_parton_delta_mae(
        parton_cmp,
        label_a=label_a,
        label_b=label_b,
        outpath=os.path.join(outdir, f"parton_delta_mae_{label_a}_vs_{label_b}.png"),
    )

    winbar_variable_delta(
        var_cmp,
        label_a=label_a,
        label_b=label_b,
        metric="rel_delta_mse",
        title_prefix="relative ΔMSE",
        outpath=os.path.join(outdir, f"variables_winbar_rel_delta_mse_{label_a}_vs_{label_b}.png"),
    )

    plot_topk_variable_delta(
        var_cmp,
        label_a=label_a,
        label_b=label_b,
        metric="rel_delta_mse",
        top_k=10,
        ascending=True,
        outpath=os.path.join(outdir, f"variables_rel_delta_mse_top10_best_{label_a}_vs_{label_b}.png"),
        include_parton_in_label=True,
    )

    plot_topk_variable_delta(
        var_cmp,
        label_a=label_a,
        label_b=label_b,
        metric="rel_delta_mse",
        top_k=10,
        ascending=False,
        outpath=os.path.join(outdir, f"variables_rel_delta_mse_top10_worst_{label_a}_vs_{label_b}.png"),
        include_parton_in_label=True,
    )

    # relative metrics by parton
    plot_parton_metric_bar(
        parton_cmp, label_a, label_b,
        metric_base="rel_mae_mean",
        ylabel="relative MAE (MAE / mean(|truth|))",
        outpath=os.path.join(outdir, f"parton_rel_mae_{label_a}_vs_{label_b}.png"),
    )

    plot_parton_metric_bar(
        parton_cmp, label_a, label_b,
        metric_base="rel_mse_mean",
        ylabel="relative MSE (MSE / mean(truth^2))",
        outpath=os.path.join(outdir, f"parton_rel_mse_{label_a}_vs_{label_b}.png"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from CV comparison csvs."
    )
    parser.add_argument("--parton-csv", required=True,
                        help="Path to parton-level comparison csv.")
    parser.add_argument("--variable-csv", required=True,
                        help="Path to variable-level comparison csv.")
    parser.add_argument("--label-a", required=True,
                        help="Label for model A (used in column names).")
    parser.add_argument("--label-b", required=True,
                        help="Label for model B (used in column names).")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for plots (defaults to csv location).")

    args = parser.parse_args()

    # one entry point: read csvs -> dump plots
    plot_from_csvs(
        parton_csv=args.parton_csv,
        variable_csv=args.variable_csv,
        label_a=args.label_a,
        label_b=args.label_b,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()