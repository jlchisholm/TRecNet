import os
from statistics import NormalDist

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def apply_thesis_style():
    # central place to lock in plot style so everything looks the same
    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 5.0),
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
            "axes.grid": False,
        }
    )


def _ensure_dir(path):
    # mkdir -p for the parent folder of a file path
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _add_atlas_label(ax, atlas_label=None):
    # add ATLAS label in figure coordinates (outside the axes box)
    if not atlas_label or not atlas_label.get("enabled", False):
        return

    fig = ax.figure
    pos = ax.get_position()
    x0, y0, x1, y1 = pos.x0, pos.y0, pos.x1, pos.y1

    pad = atlas_label.get("pad", 0.01)
    loc = atlas_label.get("loc", "upper left")
    loc_map = {
        "upper left": (x0 + pad, y1 + pad, "left", "bottom"),
        "upper right": (x1 - pad, y1 + pad, "right", "bottom"),
        "lower left": (x0 + pad, y0 - pad, "left", "top"),
        "lower right": (x1 - pad, y0 - pad, "right", "top"),
    }
    x, y, ha, va = loc_map.get(loc, loc_map["upper left"])

    fig.text(
        x,
        y,
        atlas_label.get("text", "ATLAS"),
        transform=fig.transFigure,
        ha=ha,
        va=va,
        fontsize=atlas_label.get("fontsize", 14),
        fontweight=("bold" if atlas_label.get("bold", True) else "normal"),
        color=atlas_label.get("color", "black"),
        alpha=atlas_label.get("alpha", 1.0),
    )


def _axis_label(prefix, var_spec):
    # just keeps the formatting consistent
    return f"{prefix} {var_spec['label']}".strip()


def _metric_text(label, stats):
    # quick stats box shown on residual plots
    return (
        f"{label}\n"
        f"MAE = {stats['err_mae']:.3g}\n"
        f"bias = {stats['err_bias']:.3g}\n"
        f"$\\sigma_{{68}}$ = {stats['err_sigma68']:.3g}"
    )


def plot_truth_reco_density(truth, reco, var_spec, label, outpath, atlas_label=None):
    # truth vs reco density 
    apply_thesis_style()
    outpath = _ensure_dir(outpath)

    truth = np.asarray(truth).ravel()
    reco = np.asarray(reco).ravel()

    fig, ax = plt.subplots()
    plot_range = var_spec["truth_range"]

    hb = ax.hexbin(
        truth,
        reco,
        gridsize=55,
        extent=(plot_range[0], plot_range[1], plot_range[0], plot_range[1]),
        mincnt=1,
        cmap="Blues",
    )
    fig.colorbar(hb, ax=ax, label="Counts")

    # y=x guide line
    ax.plot(plot_range, plot_range, color="black", linestyle="--", linewidth=1.3)
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)

    ax.set_xlabel(_axis_label("Parton-level", var_spec))
    ax.set_ylabel(_axis_label("Reco-level", var_spec))
    ax.set_title(f"{label}: parton vs reco")

    # have N on the plot just so we know it isn't secretly empty
    ax.text(
        0.03,
        0.97,
        f"N = {truth.size:,}\nall folds pooled",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    _add_atlas_label(ax, atlas_label)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_residual_overlay(
    err_a,
    err_b,
    stats_a,
    stats_b,
    var_spec,
    label_a,
    label_b,
    color_a,
    color_b,
    outpath,
    atlas_label=None,
):
    # overlay residual/resolution distributions for two models (A vs B)
    apply_thesis_style()
    outpath = _ensure_dir(outpath)

    fig, ax = plt.subplots()
    hist_range = var_spec["residual_range"]
    nbins = int(var_spec.get("residual_nbins", 30))

    # using step hist so overlay is readable
    ax.hist(
        err_a,
        bins=nbins,
        range=hist_range,
        histtype="step",
        density=True,
        color=color_a,
        label=label_a,
    )
    ax.hist(
        err_b,
        bins=nbins,
        range=hist_range,
        histtype="step",
        density=True,
        color=color_b,
        label=label_b,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)

    # label depends on whether we're in fractional pt space or plain residuals
    if var_spec["error_mode"] == "resolution":
        xlabel = f"{var_spec['label']} Resolution"
    else:
        xlabel = f"{var_spec['label']} Residuals"

    ax.set_xlim(hist_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events (Normalized)")
    ax.set_title(var_spec["variable"])
    ax.legend(loc="upper right")

    # little stat boxes in plot coords
    ax.text(
        0.03,
        0.97,
        _metric_text(label_a, stats_a),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=color_a,
    )
    ax.text(
        0.97,
        0.97,
        _metric_text(label_b, stats_b),
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=color_b,
    )

    _add_atlas_label(ax, atlas_label)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_residual_single(err, stats, var_spec, label, color, outpath, atlas_label=None):
    # same as overlay but for a single model 
    apply_thesis_style()
    outpath = _ensure_dir(outpath)

    fig, ax = plt.subplots()
    ax.hist(
        err,
        bins=int(var_spec.get("residual_nbins", 30)),
        range=var_spec["residual_range"],
        histtype="step",
        density=True,
        color=color,
        label=label,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(var_spec["residual_range"])

    if var_spec["error_mode"] == "resolution":
        xlabel = f"{var_spec['label']} Resolution"
    else:
        xlabel = f"{var_spec['label']} Residuals"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events (Normalized)")
    ax.set_title(f"{label}: {var_spec['variable']}")
    ax.legend(loc="upper right")

    ax.text(
        0.03,
        0.97,
        _metric_text(label, stats),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=color,
    )

    _add_atlas_label(ax, atlas_label)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bar(
    compare_df,
    label_a,
    label_b,
    metric_base,
    row_label_col,
    outpath,
    ylabel=None,
    title=None,
    color_a=None,
    color_b=None,
    atlas_label=None,
):
    # generic bar plot for comparing one metric across a list of rows (key vars, partons, etc)
    apply_thesis_style()
    outpath = _ensure_dir(outpath)

    # metric_base is "..._mean" and std is "..._std"
    metric_std = metric_base.replace("_mean", "_std")
    col_a = f"{metric_base}_{label_a}"
    col_b = f"{metric_base}_{label_b}"
    err_a = f"{metric_std}_{label_a}"
    err_b = f"{metric_std}_{label_b}"

    labels = compare_df[row_label_col].tolist()
    x = np.arange(len(labels))

    # widen figure for lots of labels so it doesn't turn into a barcode
    fig, ax = plt.subplots(figsize=(max(6.5, 1.2 * len(labels)), 5.0))
    ax.bar(
        x - 0.18,
        compare_df[col_a].to_numpy(),
        width=0.36,
        color=color_a,
        label=label_a,
        yerr=compare_df[err_a].to_numpy() if err_a in compare_df.columns else None,
        capsize=3,
    )
    ax.bar(
        x + 0.18,
        compare_df[col_b].to_numpy(),
        width=0.36,
        color=color_b,
        label=label_b,
        yerr=compare_df[err_b].to_numpy() if err_b in compare_df.columns else None,
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel or metric_base)
    ax.set_title(title or metric_base)
    ax.legend(loc="upper right")

    _add_atlas_label(ax, atlas_label)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_qq(residual, var_spec, label, outpath, atlas_label=None):
    # Q-Q plot vs a normal distribution 
    apply_thesis_style()
    outpath = _ensure_dir(outpath)

    residual = np.asarray(residual).ravel()
    residual = residual[np.isfinite(residual)]

    # this can get huge, so cap it to keep runtime sane
    if residual.size > 200000:
        rng = np.random.default_rng(12345)
        residual = rng.choice(residual, size=200000, replace=False)

    residual = np.sort(residual)

    n = residual.size
    probs = (np.arange(1, n + 1) - 0.5) / n

    # try scipy for speed; fallback to stdlib if scipy isn't installed
    try:
        from scipy.stats import norm
        theoretical = norm.ppf(probs)
    except Exception:
        theoretical = np.array([NormalDist().inv_cdf(p) for p in probs])

    fig, ax = plt.subplots()
    ax.scatter(theoretical, residual, s=8, color="black", alpha=0.7)

    # diagonal guide line
    diag_min = min(theoretical.min(), residual.min())
    diag_max = max(theoretical.max(), residual.max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="tab:red")

    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel(f"{var_spec['label']} error quantiles")
    ax.set_title(f"{label}: Q-Q")

    _add_atlas_label(ax, atlas_label)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)