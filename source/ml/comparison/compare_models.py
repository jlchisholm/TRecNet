##########################################################################
#                                                                        #
#  compare_models.py                                                     #
#  Author: Tommy Lubomirski (refactor by GPT)                            #
#  Updated: Oct.1/25                                                     #
#  Adapted from: run_validation.py                                       #
#                                                                        #
#  Compares two trained TRecNet models on given data using the same      #
#  logic as run_validation (via Predictor.get_*_pred_dics).              #
#  Saves plots + metrics into                                            #
#    TRecNet/model_comparisons/compare_{model1}_{model2}/                #
#                                                                        #
# Github Copilot was used as a linter, and for debugging                 #                                                                     #
##########################################################################

import os, sys
import argparse
import json
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1") 

from source.ml.Models.blocks import set_encoder, transformer_blocks, objwise, pooling
from source.ml.Predictions import Predictor
from source.ml.TRecNet_Model import TRecNet_Model

from source.ml.paths import resolve_model_dir, ensure_dir


import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sb

#from sklearn.metrics import mean_absolute_error, mean_squared_error, multilabel_confusion_matrix



def fprintf(path, text):
    '''write to log file'''
    with open(path, 'a') as f:
        f.write(text + ('\n' if not text.endswith('\n') else ""))



def compute_basic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    eps = 1e-12
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    scale_mae = float(np.mean(np.abs(y_true)) + eps)
    scale_mse = float(np.mean(y_true ** 2) + eps)

    rel_mae = float(mae / scale_mae)
    rel_mse = float(mse / scale_mse)
    return mae, mse, rel_mae, rel_mse

def plot_pred_vs_truth(var, truths, preds1, preds2, label1, label2, save_path):

    def clean_label(label):
        label = label.split('_')
        return label[2]
    truths = np.asarray(truths).ravel()
    preds1 = np.asarray(preds1).ravel()
    preds2 = np.asarray(preds2).ravel()
    label1 = clean_label(label1)
    label2 = clean_label(label2)

    nbins = 40

    vmin, vmax = truths.min(), truths.max()
    edges = np.linspace(vmin, vmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]

    # get the histograms binned the same
    h_truth, _ = np.histogram(truths, bins=edges)
    h_p1,    _ = np.histogram(preds1, bins=edges)
    h_p2,    _ = np.histogram(preds2, bins=edges)

    # calculate the accuracy per bin
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = h_p1 / h_truth
        r2 = h_p2 / h_truth
    # ignore bins with no predictions
    mask = h_truth > 0
    r1[~mask] = np.nan
    r2[~mask] = np.nan

    # PLOTTING CODE
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # top plot: distributions
    ax_top.hist(truths, bins=edges, histtype="step", label="Truth")
    ax_top.hist(preds1, bins=edges, histtype="step", label=label1)
    ax_top.hist(preds2, bins=edges, histtype="step", label=label2)
    ax_top.set_ylabel("Events")
    ax_top.legend(loc="best")
    ax_top.tick_params(labelbottom=False)

    # bottom plot: per-bin accuracy
    ax_bot.axhline(1.0, color="black", lw=0.8)
    # plot as step-lines so they align with the histogram bins
    ax_bot.step(centers, r1, where="mid", label=f"{label1}", linewidth=1.4)
    ax_bot.step(centers, r2, where="mid", label=f"{label2}", linewidth=1.4)
    ax_bot.set_xlabel(var)
    ax_bot.set_ylabel("pred/truth")
    ax_bot.legend(loc="best")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")
    plt.close(fig)

def plot_residuals(var, truths, preds, label, save_path):
    truths = np.asarray(truths).ravel()
    preds = np.asarray(preds).ravel()
    finite = np.isfinite(truths) & np.isfinite(preds)

    res = preds[finite] - truths[finite]

    
    nbins = 40
    res_range = (np.nanpercentile(res, 1), np.nanpercentile(res, 99))
    plt.figure(figsize = (7,5))
    plt.hist(res, bins = nbins, range = res_range, histtype = 'step', label = f'{label}')
    plt.xlabel(f'{var} residuals')
    plt.ylabel('Events')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150)
    print(f'Saved to {save_path}')
    plt.close()


def write_metrics_table(metrics_rows, save_csv):

    df = pd.DataFrame(
        metrics_rows,
        columns=[
            "scope","variable",
            "mae_model1","mae_model2","delta_mae",
            "mse_model1","mse_model2","delta_mse",
            "rel_mae_model1","rel_mae_model2","delta_rel_mae",
            "rel_mse_model1","rel_mse_model2","delta_rel_mse",
        ],
    )
    df.sort_values(["scope", "variable"], inplace = True)
    df.to_csv(save_csv, index = False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare two models on TEST data.")
    parser.add_argument("--model1", required=True, help="MODEL_ID for model #1")
    parser.add_argument("--model2", required=True, help="MODEL_ID for model #2")
    parser.add_argument("--data",   default="/data/tommylub/h5_files/ttbb_603192_mc20d_fullsim_10jets_b1b2_nom.h5", help="Path to TEST H5 file")
    parser.add_argument("--split",  default="test", choices=["test","val"], help=" test or val (needed to use run_validation.py functions)")
    parser.add_argument("--outroot", default=os.path.join(ROOT, "model_comparisons"), help="Folder dir for comparisons")
    args = parser.parse_args()

    mid1, mid2 = args.model1, args.model2
    # tag1, tag2 = sanitize_id(mid1), sanitize_id(mid2)
    tag1, tag2 = mid1, mid2


    # mnake output dirs
    comp_dir = ensure_dir(os.path.join(args.outroot, f"compare_{tag1}_{tag2}"))
    scaled_dir = ensure_dir(os.path.join(comp_dir, "scaled"))
    orig_dir   = ensure_dir(os.path.join(comp_dir, "original"))
    resid_dir  = ensure_dir(os.path.join(comp_dir, "residuals"))
    log_path   = os.path.join(comp_dir, "compare.log")

    # print to log
    fprintf(log_path, f"COMPARING:\n  model1: {mid1}\n  model2: {mid2}\n  data: {args.data}")
    
    # See if we can find the models
    try:
        m1_dir = resolve_model_dir(mid1)
        m2_dir = resolve_model_dir(mid2)
        fprintf(log_path, f"Resolved dirs:\n  m1_dir: {m1_dir}\n  m2_dir: {m2_dir}")
    except Exception as e:
        fprintf(log_path, f" resolve_model_dir failed")
        return 
    

    # Load models
    model1 = TRecNet_Model()
    model2 = TRecNet_Model()
    model1.load(mid1)
    model2.load(mid2)

    pred = Predictor()

    # Get predictions (mirrors run_validation flow)
    s_pred1, s_true1, o_pred1, o_true1 = pred.get_scaled_and_origscale_pred_dics(model1, args.data, args.split)
    s_pred2, s_true2, o_pred2, o_true2 = pred.get_scaled_and_origscale_pred_dics(model2, args.data, args.split)


    # lets get keys, we can get unique keys by intersecting the sets 
    # (same way you get feature names in sklearn)

    scaled_keys = sorted(set(s_pred1.keys()) & set(s_true1.keys()) & set(s_pred2.keys()) & set(s_true2.keys()) )
    orig_keys = sorted(set(o_pred1.keys()) & set(o_true1.keys()) & set(o_pred2.keys()) & set(o_true2.keys()) )

    # metrics
    rows = []

    for var in scaled_keys:
        y = s_true1[var]
        p1 = s_pred1[var]
        p2 = s_pred2[var]

        # plot hists

        plot_pred_vs_truth(var, y, p1, p2,
            label1 = tag1, label2 = tag2,
            save_path=os.path.join(scaled_dir,f"{var}.png"))

        # residuals
        plot_residuals(var, y, p1, label = f'{tag1} scaled',
            save_path=os.path.join(resid_dir, f"{var}_res_{tag1}_scaled.png"))
        plot_residuals(var, y, p2, label = f'{tag2} scaled',
            save_path=os.path.join(resid_dir, f"{var}_res_{tag2}_scaled.png"))

        mae1, mse1, rmae1, rmse1 = compute_basic_metrics(y, p1)
        mae2, mse2, rmae2, rmse2 = compute_basic_metrics(y, p2)

        rows.append([
            'scaled', var,
            mae1, mae2, mae2-mae1,
            mse1, mse2, mse2-mse1,
            rmae1, rmae2, rmae2-rmae1,
            rmse1, rmse2, rmse2-rmse1,
        ])
    for var in orig_keys:
        y = o_true1[var]
        p1 = o_pred1[var]
        p2 = o_pred2[var]

        # plot hists

        plot_pred_vs_truth(var, y, p1, p2,
            label1 = tag1, label2 = tag2,
            save_path=os.path.join(orig_dir,f"{var}.png"))

        # residuals
        plot_residuals(var, y, p1, label = f'{tag1} original',
            save_path=os.path.join(resid_dir, f"{var}_res_{tag1}_original.png"))
        plot_residuals(var, y, p2, label = f'{tag2} original',
            save_path=os.path.join(resid_dir, f"{var}_res_{tag2}_original.png"))

        mae1, mse1, rmae1, rmse1 = compute_basic_metrics(y, p1)
        mae2, mse2, rmae2, rmse2 = compute_basic_metrics(y, p2)

        rows.append([
            'original', var,
            mae1, mae2, mae2-mae1,
            mse1, mse2, mse2-mse1,
            rmae1, rmae2, rmae2-rmae1,
            rmse1, rmse2, rmse2-rmse1,
        ])
    metrics_csv = os.path.join(comp_dir, "metrics.csv")
    df = write_metrics_table(rows, metrics_csv)
    fprintf(log_path, "Comparison complete :)")

    def _block(scope):
        sub = df[df["scope"]==scope]
        if sub.empty: return []
        lines = []
        # best_mae = (sub["delta_mae"].sum() < 0)
        # best_mse = (sub["delta_mse"].sum() < 0)
        lines.append(f"{scope.title()} — sigma_delta_MAE = {sub['delta_mae'].sum():.6g}, sigma_delta_MSE = {sub['delta_mse'].sum():.6g}  (negative favours model2)")
        lines.append(f"  MAE winners: {(sub[['variable','delta_mae']].sort_values('delta_mae').head(5)).to_string(index=False)}")
        lines.append(f"  MSE winners: {(sub[['variable','delta_mse']].sort_values('delta_mse').head(5)).to_string(index=False)}")
        lines.append(
            f"{scope.title()} — "
            f"ΣΔMAE={sub['delta_mae'].sum():.6g}, "
            f"ΣΔMSE={sub['delta_mse'].sum():.6g}, "
            f"ΣΔrelMAE={sub['delta_rel_mae'].sum():.6g}, "
            f"ΣΔrelMSE={sub['delta_rel_mse'].sum():.6g} "
            "(negative favours model2)"
        )
        return lines

    summary_path = os.path.join(comp_dir, "summary.txt")
    fprintf(summary_path, f"Model1: {mid1}\nModel2: {mid2}\nData: {args.data}\nSplit: {args.split}\n")
    for ln in _block("scaled"): fprintf(summary_path, ln)
    for ln in _block("original"): fprintf(summary_path, ln)

if __name__ == "__main__":
    main()