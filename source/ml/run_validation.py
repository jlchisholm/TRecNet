##########################################################################
#                                                                        #
#  run_validation.py                                                     #
#  Author: Jenna Chisholm                                                #
#  Updated: Sept.8/25                                                    #
#                                                                        #
#  Uses the specified trained TRecNet model and dataset to make          #
#  predictions. Preliminary plots of these predictions compared to truth #
#  are saved in the appropriate trained model directory. The intended    #
#  use is not for testing the model, but validating and debugging the    #
#  training.                                                             # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

from argparse import ArgumentParser
import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sb
import tracemalloc
tracemalloc.start()

# All imports relative to TRecNet/ directory
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1") 
from source.ml.Predictions import Predictor
from source.ml.TRecNet_Model import TRecNet_Model
from source.ml.Models.blocks import set_encoder, transformer_blocks, objwise, pooling
from source.ml.paths import resolve_model_dir



def plot_pred_vs_truth(model_name, var, preds, truths, save_loc):
        
    # Some settings
    ticks_size = 14
    axis_size = 16
    legend_size = 14
    nbins = 30
    max_val = max(max(preds), max(truths))
    min_val = min(min(preds), min(truths))
    var_label = var
    
    # Make figure and fill hisotgram
    plt.figure()
    plt.hist(truths, bins=nbins, range=(min_val, max_val), histtype='step',label='Truth')
    if model_name == 'JetPretrainer' or model_name == 'bbPretrainer': 
        plt.hist(np.array(preds).round(), bins=nbins, range=(min_val, max_val), histtype='step',label='reco (rounded)')
    else:
        plt.hist(preds, bins=nbins, range=(min_val, max_val), histtype='step',label='Prediction')
        
    # Other settings
    plt.legend(fontsize=legend_size, loc='lower right')
    plt.xlabel(var_label,fontsize=axis_size)
    plt.ylabel('Events',fontsize=axis_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    outpath = os.path.join(save_loc, var + ".png")   # directory to save to
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print('Saved figure: ' + outpath)

      
def jet_cm_plot(njets, preds, truths, save_loc):
    
    # Make confusion matrix
    cms = multilabel_confusion_matrix(np.array(truths), np.array(preds).round())
    
    # Determine number of plots in each row/column, depending on njets ([0] = 0 jets, [1] = 1 jet, etc.)
    rows = [0,1,1,1,2,1,2,1,2,3,2]
    cols = [0,1,2,3,2,5,3,7,4,3,5]

    # Set up figure set
    fig, ax = plt.subplots(rows[njets],cols[njets], sharex=True, sharey=True, figsize=(10, 8))
    cbar_ax = fig.add_axes([1.02, .11, .03, .82])

    jet = 0
    for i in range(rows[njets]):
        for j in range(cols[njets]):
            cm = cms[jet]
            group_names = ['True Negative','False Positive','False Negative','True Positive']
            group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cm[0].flatten()/np.sum(cm[0])]
            group_percentages.extend(["{0:.2%}".format(value) for value in cm[1].flatten()/np.sum(cm[1])])
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            normed = [cm[0]*100/np.sum(cm[0]),cm[1]*100/np.sum(cm[1])]
            sub = sb.heatmap(normed,vmin=0, vmax=100, cmap='YlGnBu',fmt='',annot=labels, cbar=True, cbar_ax=cbar_ax,square=True, ax=ax[i,j])
            ax[i,j].set_title('Jet 1'+str(jet+1))
            ax[i,j].tick_params(left=False,bottom=False) 
            jet+=1
            #print('j'+str(jet+1)+' Accuracy: ',accuracy_score(true_unscaled[:,jet],predictions_unscaled.round()[:,jet]))
            #print('j'+str(jet+1)+' Precision: ',precision_score(true_unscaled[:,jet],predictions_unscaled.round()[:,jet]))


    fig.supxlabel('Reco')
    fig.supylabel('Truth')

    fig.tight_layout()
    fig.savefig(save_loc+"JetPretrain_Normalized_CM",bbox_inches='tight')


def make_plots(model, scale_pred_dic, scale_true_dic, origscale_pred_dic, origscale_true_dic):
    # finds the trained run directory via resolve_model_dir(model.model_id)
    model_dir = resolve_model_dir(model.model_id)

    # save dirs for validation plots
    # creates two dedicated output dirs
    # plots/val/scaled
    # plots/val/original
    save_loc_scaled   = os.path.join(model_dir, "plots", "val", "scaled")
    save_loc_original = os.path.join(model_dir, "plots", "val", "original")
    os.makedirs(save_loc_scaled, exist_ok=True)
    os.makedirs(save_loc_original, exist_ok=True)

    model_name = getattr(model, 'model_name', model.model_id)

    # Scaled Plots
    for key in scale_pred_dic.keys():
        plot_pred_vs_truth(model_name, key, scale_pred_dic[key], scale_true_dic[key], save_loc_scaled)

    # Original Scale Plots
    for key in origscale_pred_dic.keys():
        plot_pred_vs_truth(model_name, key, origscale_pred_dic[key], origscale_true_dic[key], save_loc_original)

    # Jet CM Plot (For JetPretrainer)
    if model_name == 'JetPretrainer':
        jet_cm_plot(model.njets, scale_pred_dic, scale_true_dic, save_loc_scaled)
        #jet_cm_plot(preds_origscale_dic, true_origscale_dic, save_loc_original)    




### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-i', '--model_id', help="ID of the model.", type=str, required=True)
    parser.add_argument('-d', '--train_data', help="Path and file name for the training data to be used.", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    
    # Load the model
    model = TRecNet_Model()
    model.load(args.model_id)

    # Test the model
    print('Beginning validation for '+args.model_id+'...')
    predictor = Predictor()
    scale_pred_dic, scale_true_dic, origscale_pred_dic, origscale_true_dic = predictor.get_scaled_and_origscale_pred_dics(model, args.train_data, 'val')
    make_plots(model, scale_pred_dic, scale_true_dic, origscale_pred_dic, origscale_true_dic)
    
    model_dir = resolve_model_dir(model.model_id)
    results_save_path = os.path.join(model_dir, 'validation', 'Val_Results.root')
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    results_file = uproot.recreate(results_save_path)

    results_file["reco"] = origscale_pred_dic
    results_file["reco_scaled"] = scale_pred_dic

    results_file["parton"] = origscale_true_dic
    results_file["parton_scaled"] = scale_true_dic

    print('Results saved in %s.' % results_save_path)


    print('Validation complete! :)')