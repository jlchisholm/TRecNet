import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# columns that define "what row is this" (vs actual metrics)
IDENTITY_COLUMNS = {
    "fold",
    "variable",
    "parton",
    "observable",
    "error_mode",
}

# event counters (also not treated as "metrics" for mean/std loops)
COUNT_COLUMNS = {"n_events"}


def _finite_truth_pred(y_true, y_pred):
    # flatten + keep only finite entries in both arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def compute_basic_metrics(y_true, y_pred):
    # legacy metrics on raw reco-truth residuals (not wrapped, not pt-resolution)
    y_true, y_pred = _finite_truth_pred(y_true, y_pred)
    if y_true.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    diff = y_pred - y_true
    eps = 1e-8  # just to avoid dividing by zero on weird edge cases

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    # normalizations for dimensionless versions
    scale_mae = float(np.mean(np.abs(y_true)) + eps)
    scale_mse = float(np.mean(y_true ** 2) + eps)

    rel_mae = float(mae / scale_mae)
    rel_mse = float(mse / scale_mse)

    return mae, mse, rel_mae, rel_mse


def parse_variable_name(base_var):
    # split "th_pt" -> ("th", "pt"), etc
    parts = base_var.split("_", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def wrap_delta_phi(dphi):
    # wrap to (-pi, pi] so phi residuals dont look insane at the branch cut
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def infer_error_mode(base_var):
    # pick the error definition for each observable
    _, observable = parse_variable_name(base_var)
    if observable == "pt":
        return "resolution"        # (reco - truth) / truth
    if observable == "phi":
        return "wrapped_residual"  # wrap reco-truth
    return "residual"             # plain reco-truth


def build_error_array(y_true, y_pred, base_var):
    # convert truth/reco into the error space used for err_* metrics
    y_true, y_pred = _finite_truth_pred(y_true, y_pred)
    error_mode = infer_error_mode(base_var)

    if y_true.size == 0:
        return y_true, y_pred, np.array([], dtype=float)

    if error_mode == "resolution":
        # for pt: fractional resolution, but skip truth=0 so we dont blow up
        err = np.divide(
            y_pred - y_true,
            y_true,
            out=np.full_like(y_true, np.nan, dtype=float),
            where=y_true != 0,
        )
    elif error_mode == "wrapped_residual":
        err = wrap_delta_phi(y_pred - y_true)
    else:
        err = y_pred - y_true

    # only keep finite errors 
    mask = np.isfinite(err)
    return y_true[mask], y_pred[mask], err[mask]


def compute_error_metrics(err):
    # all the stats we want on err 
    err = np.asarray(err).ravel()
    err = err[np.isfinite(err)]

    if err.size == 0:
        # keep schema consistent even if there's no data
        return {
            "err_n_events": 0,
            "err_mae": np.nan,
            "err_mse": np.nan,
            "err_rmse": np.nan,
            "err_bias": np.nan,
            "err_median_abs": np.nan,
            "err_q16": np.nan,
            "err_q50": np.nan,
            "err_q84": np.nan,
            "err_sigma68": np.nan,
            "err_q025": np.nan,
            "err_q975": np.nan,
            "err_tail_frac_3sigma": np.nan,
        }

    abs_err = np.abs(err)

    # quantiles are way more robust than just std when we have tails
    q16, q50, q84 = np.quantile(err, [0.16, 0.50, 0.84])
    q025, q975 = np.quantile(err, [0.025, 0.975])

    # sigma68 = half-width of the central 68% interval
    sigma68 = 0.5 * (q84 - q16)

    # simple tail metric: how many events are beyond 3*sigma68
    if sigma68 > 0:
        tail_frac = float(np.mean(abs_err > 3.0 * sigma68))
    else:
        tail_frac = 0.0

    return {
        "err_n_events": int(err.size),
        "err_mae": float(np.mean(abs_err)),
        "err_mse": float(np.mean(err ** 2)),
        "err_rmse": float(np.sqrt(np.mean(err ** 2))),
        "err_bias": float(np.mean(err)),
        "err_median_abs": float(np.median(abs_err)),
        "err_q16": float(q16),
        "err_q50": float(q50),
        "err_q84": float(q84),
        "err_sigma68": float(sigma68),
        "err_q025": float(q025),
        "err_q975": float(q975),
        "err_tail_frac_3sigma": tail_frac,
    }


def _discover_base_vars(fold_to_df):
    # find base vars that exist in every fold, so comparisons are stable
    common = None
    for _, df in fold_to_df.items():
        truth_cols = [c for c in df.columns if c.startswith("truth_")]
        reco_cols  = [c for c in df.columns if c.startswith("reco_")]
        truth_bases = {c[len("truth_"):] for c in truth_cols}
        reco_bases  = {c[len("reco_"):] for c in reco_cols}
        bases = truth_bases & reco_bases
        common = bases if common is None else (common & bases)

    base_vars = sorted(common) if common is not None else []
    logger.info("Found %d common base vars across folds.", len(base_vars))
    return base_vars


def compute_per_fold_metrics(fold_to_df, base_vars=None, require_both_truth_reco=True):
    # main worker: fold_to_df -> long table with one row per (fold, variable)
    if not fold_to_df:
        raise RuntimeError("compute_per_fold_metrics called with empty fold_to_df.")

    if base_vars is None:
        base_vars = _discover_base_vars(fold_to_df)

    rows = []

    # sort folds just so outputs dont jitter between runs
    try:
        fold_items = sorted(fold_to_df.items(), key=lambda kv: kv[0])
    except Exception:
        fold_items = list(fold_to_df.items())

    for fold, df in fold_items:
        for base_var in base_vars:
            truth_col = f"truth_{base_var}"
            reco_col = f"reco_{base_var}"

            # if we said "require", then fail loud when something is missing
            if require_both_truth_reco and (truth_col not in df.columns or reco_col not in df.columns):
                raise RuntimeError(f"Missing truth/reco columns for {base_var} in fold {fold}.")
            if truth_col not in df.columns or reco_col not in df.columns:
                continue

            y_true = df[truth_col].to_numpy()
            y_pred = df[reco_col].to_numpy()

            # just raw count of finite entries 
            n_events = int(np.sum(np.isfinite(y_true) & np.isfinite(y_pred)))

            # legacy raw residual metrics
            mae, mse, rel_mae, rel_mse = compute_basic_metrics(y_true, y_pred)

            # "error-mode" metrics (pt resolution / wrapped phi / etc)
            _, _, err = build_error_array(y_true, y_pred, base_var)
            error_metrics = compute_error_metrics(err)

            parton, observable = parse_variable_name(base_var)

            rows.append(
                {
                    "fold": fold,
                    "variable": base_var,
                    "parton": parton,
                    "observable": observable,
                    "error_mode": infer_error_mode(base_var),
                    "n_events": n_events,
                    "mae": mae,
                    "mse": mse,
                    "rel_mae": rel_mae,
                    "rel_mse": rel_mse,
                    **error_metrics,
                }
            )

    return pd.DataFrame(rows)


def _metric_columns(df, metric_cols=None):
    # decide what columns to run mean/std over
    if metric_cols is not None:
        return list(metric_cols)

    columns = []
    for col in df.columns:
        if col in IDENTITY_COLUMNS or col in COUNT_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    return columns


def _filtered_metrics_df(per_fold_df, variables=None, observables=None):
    # convenience filter so we can do "pt-only" or "key vars only" summaries
    df = per_fold_df.copy()

    if variables is not None:
        df = df[df["variable"].isin(variables)]
    if observables is not None:
        df = df[df["observable"].isin(observables)]

    return df


def aggregate_metrics_by_variable(per_fold_df, metric_cols=None):
    # fold-aggregated stats for each variable (mean/std across folds)
    metrics = _metric_columns(per_fold_df, metric_cols=metric_cols)
    rows = []

    for variable, group in per_fold_df.groupby("variable", sort=False):
        # identity columns (parton/observable should be consistent within a base var)
        row = {
            "variable": variable,
            "parton": group["parton"].iloc[0],
            "observable": group["observable"].iloc[0],
            "error_mode": group["error_mode"].iloc[0],
            "n_folds": int(group["fold"].nunique()),
            "n_events_total": int(group["n_events"].sum()),
            "err_n_events_total": int(group["err_n_events"].sum()),
        }
        for metric in metrics:
            values = group[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=0))
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["parton", "observable", "variable"])
        .reset_index(drop=True)
    )


def aggregate_metrics_by_parton(per_fold_df, metric_cols=None, observables=None):
    # parton summary is a bit tricky: do fold-first so std actually means "spread across folds"
    filtered = _filtered_metrics_df(per_fold_df, observables=observables)
    metrics = _metric_columns(filtered, metric_cols=metric_cols)
    rows = []

    # first average within each fold for each parton (across variables)
    fold_parton_rows = []
    for (fold, parton), g in filtered.groupby(["fold", "parton"], sort=False):
        r = {
            "fold": fold,
            "parton": parton,
            "n_variables": int(g["variable"].nunique()),
            "n_events_total": int(g["n_events"].sum()),
            "err_n_events_total": int(g["err_n_events"].sum()),
        }
        for metric in metrics:
            r[metric] = float(np.nanmean(g[metric].to_numpy()))
        fold_parton_rows.append(r)

    fold_parton_df = pd.DataFrame(fold_parton_rows)

    # next aggregate across folds
    for parton, group in fold_parton_df.groupby("parton", sort=False):
        row = {
            "parton": parton,
            "n_variables": int(filtered[filtered["parton"] == parton]["variable"].nunique()),
            "n_folds": int(group["fold"].nunique()),
            "n_events_total": int(filtered[filtered["parton"] == parton]["n_events"].sum()),
            "err_n_events_total": int(filtered[filtered["parton"] == parton]["err_n_events"].sum()),
        }
        if observables is not None:
            # keep track of what we aggregated over (mostly for pt-only summaries)
            row["observables"] = ",".join(observables)

        for metric in metrics:
            values = group[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=0))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("parton").reset_index(drop=True)


def aggregate_metrics_global(per_fold_df, variables=None, observables=None, metric_cols=None):
    # one-row global summary (again fold-first so the std means something)
    filtered = _filtered_metrics_df(
        per_fold_df,
        variables=variables,
        observables=observables,
    )
    metrics = _metric_columns(filtered, metric_cols=metric_cols)

    fold_rows = []
    for fold, group in filtered.groupby("fold", sort=False):
        # average inside each fold first
        row = {
            "fold": fold,
            "n_variables": int(group["variable"].nunique()),
            "n_events_total": int(group["n_events"].sum()),
            "err_n_events_total": int(group["err_n_events"].sum()),
        }
        for metric in metrics:
            row[metric] = float(np.nanmean(group[metric].to_numpy()))
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        # keep output schema stable even if we filtered everything away
        return pd.DataFrame(
            [
                {
                    "summary": "global",
                    "n_folds": 0,
                    "n_variables": 0,
                    "n_events_total": 0,
                    "err_n_events_total": 0,
                }
            ]
        )

    summary = {
        "summary": "global",
        "n_folds": int(fold_df["fold"].nunique()),
        "n_variables": int(filtered["variable"].nunique()),
        "n_events_total": int(filtered["n_events"].sum()),
        "err_n_events_total": int(filtered["err_n_events"].sum()),
    }
    if variables is not None:
        summary["variables"] = ",".join(variables)
    if observables is not None:
        summary["observables"] = ",".join(observables)

    for metric in metrics:
        values = fold_df[metric].to_numpy()
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values, ddof=0))

    return pd.DataFrame([summary])