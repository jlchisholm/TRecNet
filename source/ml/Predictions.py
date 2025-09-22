#########################################################################
#                                                                       #
#  TRecNet.py.                                                          #
#  Author: Jenna Chisholm                                               #
#  Updated: Sept.8/25                                                   #
#                                                                       #
#  Defines classes and functions relevant for using TRecNet.            # 
#                                                                       #
#  Thoughts for improvements: Have X and Y keys as input variables?     #
#                                                                       #
#########################################################################


import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import uproot
from sklearn.model_selection import train_test_split
from tensorflow import keras
#from clr_callback import * 

from MLUtil import *
from Scaler import Scaler
from InfoGrabber import InfoGrabber

        
        
class Prediction:
    
    def __init__(self, model, data_file, mode):
        """
        Initializes a prediction object, which will hold one set of predictions from TRecNet.
        
            Parameters:
                model (TRecNet_model): Model to be used in the predictions.
                data_file (str): Name (including path) of the h5 dataset to predict on.
                mode (str): Identifies what type of predictions are being done ('val','test', or 'data').
                
            Attributes:
                pred_scaled (np.array): Predictions directly out of TRecNet (i.e. maxmean-scaled).
                true_scaled (np.array): Truth in the same format as predictions (i.e. maxmean-scaled).
                pred_scaled_dic (dictionary): Dictionary of predictions from TRecNet (i.e. maxmean-scaled).
                true_scaled_dic (dictionary): Dictionary of truth (i.e. maxmean-scaled).
                pred_origscale_dic (dictionary): Dictionary of predictions from TRecNet for the original (not scaled) variables.
                true_origscale_dic (dictionary): Dictionary of truth for the original (not scaled) variables.
                Y_scaled_keys (list of str): Keys that directly relate to the predicted values.
                Y_maxmean_dic (dictionary): Dictionary of maxmean values for Y variables.
        """
        
        self.model = model
        self.data_file = data_file
        self.mode = mode
        self.pred_scaled = None
        self.true_scaled = None
        self.pred_scaled_dic = None
        self.true_scaled_dic = None
        self.pred_origscale_dic = None
        self.true_origscale_dic = None
        self.Y_scaled_keys = None
        self.Y_maxmean_dic = None
        
class Predictor:
    """
    A class for running TRecNet predictions. Can be using for validation, testing, or data purposes.
    """
    
    def __init__(self):
        """
        Initializes a Predictor object.
        """    
        pass
        

    def get_scaled_predictions(self, model, data_file, mode):
        """
        Runs the pre-existing/trained TRecNet model on the given data set.
        
            Parameters:
                model (TRecNet_model): Model to be used in the predictions.
                data_file (str): Name (including path) of the h5 dataset to predict on.
                mode (str): Identifies what type of predictions are being done ('val','test', or 'data').
            
            Returns:
                prediction (Prediction): Prediction object containing necessary information regarding the prediction.
    
        """
        
        # Create prediction object to store results in
        prediction = Prediction(model, data_file, mode)

        # Create objects to use utilities
        processor = Utilities()
        grabber = InfoGrabber()
        
        # Load the things we'll need
        X_maxmean_dic, prediction.Y_maxmean_dic = processor.loadMaxMean(model.xmm_file, model.ymm_file)
        trained_model = keras.models.load_model('trained_models/'+model.model_id+'/'+model.model_id+'.keras')

        # These are the keys for what we're feeding into the pre-processing, and getting back in the end
        # X and Y variables to be used (NOTE: later have option to feed these in) OR read them in from the info file
        X_keys, Y_keys = processor.getInputKeys(model.model_v,model.n_jets,model.with_ttbar,model.extra_b_mode)

        # For val and test, we have truth values, but for data (or systematics) we don't --> treat these modes differently
        if mode=='val' or mode=='test':
            
            # Pre-process the data
            X_jets, X_other, Y_ttbar, _, prediction.Y_scaled_keys = processor.scale_and_shape(data_file, X_maxmean_dic, prediction.Y_maxmean_dic, X_keys, Y_keys, model.n_jets, -2)  # Mask value hard coded to -2

            # For validation mode, need to remove data that was used for training
            if mode=='val':
                split = grabber.get_train_val_split(model.model_id)
                _, testX_jets, _, testX_other, _, testY = train_test_split(X_jets, X_other, Y_ttbar, train_size=split)
            else:
                testX_jets, testX_other, testY =  X_jets, X_other, Y_ttbar

            # Predictions and truth BEFORE they're back to the original scale
            prediction.pred_scaled = trained_model.predict([testX_jets, testX_other])
            prediction.true_scaled = testY

        else:
            
            # Pre-process the data
            testX_jets, testX_other, _, _, prediction.Y_scaled_keys = processor.scale_and_shape(data_file, X_maxmean_dic, prediction.Y_maxmean_dic, X_keys, Y_keys, model.n_jets, -2, True)  # Mask value hard coded to -2

            # Predictions and truth BEFORE they're back to the original scale
            prediction.pred_scaled = trained_model.predict([testX_jets, testX_other])
            prediction.true_scaled = None
            
        return prediction


    def get_scaled_pred_dics(self, model, data_file, mode, prediction=None):
        """
        Runs the pre-existing/trained TRecNet model on the given data set, returning dictionaries instead of numpy arrays.
            
            Parameters:
                model (TRecNet_model): model to be used in the predictions.
                data_set (str): Name (including path) of the h5 dataset to predict on.
                mode (str): Identifies what type of predictions are being done ('val','test', or 'data').
            
            Optional Argument:
                prediction (Prediction): Prediction object for previously made prediction. If used, this will override the other arguments, and the returned dictionary will be from this Prediction.
               
            Returns:
                pred_scaled_dic (dictionary): Dictionary of predictions from TRecNet (i.e. maxmean-scaled).
                true_scaled_dic (dictionary): Dictionary of truth (i.e. maxmean-scaled) (Empty if mode != 'val' or 'test')
        """
        
        # Make predictions if not already done
        if prediction == None:
            prediction = self.get_scaled_predictions(self, model, data_file, mode)
        elif prediction.pred_scaled == None:
            print("ERROR: The prediction object you entered has no predictions in it. Exiting program.")
            sys.exit()
        
        # Convert to dictionaries
        pred_scaled_dic = {}
        true_scaled_dic = {}
        for i, key in enumerate(prediction.Y_scaled_keys):
            pred_scaled_dic[key] = prediction.pred_scaled[:,i]
            
            # Only need truth if we have it
            if mode in ['val','test']: 
                true_scaled_dic[key] = prediction.true_scaled[:,i]
            
        return pred_scaled_dic, true_scaled_dic
            
 
    def get_origscale_pred_dics(self, model, data_set, mode, prediction=None):
        """
        Runs the pre-existing/trained TRecNet model on the given data set, returning dictionaries of the original (not scaled) variables.
            
            Parameters:
                model (TRecNet_model): model to be used in the predictions.
                data_set (str): Name (including path) of the h5 dataset to predict on.
                mode (str): Identifies what type of predictions are being done ('val','test', or 'data').
                
            Optional Argument:
                prediction (Prediction): Prediction object for previously made prediction. If used, this will override the other arguments, and the returned dictionary will be from this Prediction.
            
            Returns:
                pred_origscale_dic (dictionary): Dictionary of predictions from TRecNet for the original (not scaled) variables.
                true_origscale_dic (dictionary): Dictionary of truth for the original (not scaled) variables (Empty if mode != 'val' or 'test')
        """
        
        # Make predictions if not already done
        if prediction == None:
            prediction = self.get_scaled_predictions(self, model, data_set, mode)
        elif prediction.pred_scaled == None:
            print("ERROR: The prediction object you entered has no predictions in it. Exiting program.")
            sys.exit()
        
        # Invert scaling
        scaler = Scaler()
        pred_origscale_dic = scaler.invscale_arrays(prediction.pred_scaled, prediction.Y_scaled_keys, prediction.Y_maxmean_dic)
        if mode in ['val','test']:
            true_origscale_dic = scaler.invscale_arrays(prediction.true_scaled, prediction.Y_scaled_keys, prediction.Y_maxmean_dic)
        else:
            true_origscale_dic = {}
        
        return pred_origscale_dic, true_origscale_dic
    
    
    def get_scaled_and_origscale_pred_dics(self, model, data_set, mode, prediction=None):
        """
        Runs the pre-existing/trained TRecNet model on the given data set, returning dictionaries of the original AND scaled variables.
            
            Parameters:
                model (TRecNet_model): model to be used in the predictions.
                data_set (str): Name (including path) of the h5 dataset to predict on.
                mode (str): Identifies what type of predictions are being done ('val','test', or 'data').
                
            Optional Argument:
                prediction (Prediction): Prediction object for previously made prediction. If used, this will override the other arguments, and the returned dictionary will be from this Prediction.
            
            Returns:
                pred_scaled_dic (dictionary): Dictionary of predictions from TRecNet (i.e. maxmean-scaled).
                true_scaled_dic (dictionary): Dictionary of truth (i.e. maxmean-scaled) (Empty if mode != 'val' or 'test')
                pred_origscale_dic (dictionary): Dictionary of predictions from TRecNet for the original (not scaled) variables.
                true_origscale_dic (dictionary): Dictionary of truth for the original (not scaled) variables (Empty if mode != 'val' or 'test')
        """
        
        # Make predictions if not already done
        if prediction == None:
            prediction = self.get_scaled_predictions(self, model, data_set, mode)
        elif prediction.pred_scaled == None:
            print("ERROR: The prediction object you entered has no predictions in it. Exiting program.")
            sys.exit()
            
        # Get the dictionaries from the prediction
        pred_scaled_dic, true_scaled_dic = Predictor.get_scaled_pred_dics(model, data_set, mode, prediction)
        pred_origscale_dic, true_origscale_dic = Predictor.get_origscale_pred_dics(model, data_set, mode, prediction)
        
        return pred_scaled_dic, true_scaled_dic, pred_origscale_dic, true_origscale_dic
        
        
            
                
    def predict_and_save_results(self, model, data_file, mode, save_loc, include_scaled=False):
        """
        Saves the results the model outputs from the test dataset, in a root file.

            Parameters:
                model (TRecNet_Model): Trained model object.
                data_file (str): Name (including path) of the h5 data file to predict on. 
                mode (str): Identifies what type of predictions are being done ('val','test', or 'data').
                save_loc (str): Location to save the test results.
                include_scaled (bool): Whether to also save the scaled variables to the result file.
        """
        
        # Create the save file
        data_name = data_file.split('.h5')[0]
        if 'train' in data_name and mode=='val':
            data_name = data_name.split('train')[0]+'val'
        save_name = save_loc+model.model_id+'_'+data_name+'_Results.root'
        results_file = uproot.recreate(save_name)
        
        # Make predictions
        if include_scaled:
            scale_pred_dic, scale_true_dic, pred_dic, true_dic = Predictor.get_scaled_and_origscale_pred_dics(model, data_file, mode)
        else:
            pred_dic, true_dic = Predictor.get_origscale_pred_dics(model, data_file, mode)

        # Save the results as 'reco' (pred) and 'parton' (truth) (if available)
        results_file["reco"] = pred_dic
        if include_scaled:
            results_file["reco_scaled"] = scale_pred_dic
        if mode in ['val','test']: 
            results_file["parton"] = true_dic
            if include_scaled:
                results_file["parton_scaled"] = scale_true_dic

        print('Results saved in %s.' % save_name)