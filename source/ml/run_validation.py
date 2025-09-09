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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"    # These are the GPUs visible for training
from argparse import ArgumentParser

from Predictions import Predictor
from TRecNet_Model import TRecNet_Model

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sb

import tracemalloc
tracemalloc.start()



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
    plt.savefig(save_loc+var, bbox_inches='tight')
    plt.close()
    
    print('Saved figure: '+save_loc+var)
      
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
    
    # Create directory for saving things in if it doesn't exist
    save_loc ='models/'+model.model_name+'/'+model.model_id+'/val_plots/'
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    
    # Scaled Plots
    if not os.path.exists(save_loc+'scaled/'):
        os.makedirs(save_loc+'scaled/')
    for key in scale_pred_dic.keys():
        plot_pred_vs_truth(model.model_name, key, scale_pred_dic[key], scale_true_dic[key], save_loc+'scaled/')
        
    # Original Scale Plots
    if not os.path.exists(save_loc+'original/'):
        os.makedirs(save_loc+'original/')
    for key in origscale_pred_dic.keys():
        plot_pred_vs_truth(model.model_name, key, origscale_pred_dic[key], origscale_true_dic[key], save_loc+'original/')
            
    # Jet CM Plot (For JetPretrainer)
    if model.model_name == 'JetPretrainer':
        jet_cm_plot(scale_pred_dic, scale_true_dic, save_loc+'scaled/')
        #jet_cm_plot(preds_origscale_dic, true_origscale_dic, save_loc+'original/')






### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-i', '--model_id', help="ID of the model.", type=str, required=True)
    parser.add_argument('-d', '--train_data', help="Path and file name for the training data to be used.", type=str, required=True)
    parser.add_argument('-s','--save_loc', help="Directory (including path) in which to save the results.", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    
    # Load the model
    model = TRecNet_Model()
    model.load(args.model_id)

    # Test the model
    print('Beginning predicting for '+args.model_id+'...')
    Predictor = Predictor()
    scale_pred_dic, scale_true_dic, origscale_pred_dic, origscale_true_dic = Predictor.get_scaled_and_origscale_pred_dics(model, args.train_data, 'val')
    make_plots(model, scale_pred_dic, scale_true_dic, origscale_pred_dic, origscale_true_dic)

    print('Validation complete! :)')