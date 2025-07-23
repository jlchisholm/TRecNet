##########################################################################
#                                                                        #
#  Training.py                                                           #
#  Author: Jenna Chisholm                                                #
#  Updated: Jul.23/25                                                    #
#                                                                        #
#  Defines classes and functions relevant for training and hypertuning   #
#  neural networks.                                                      # 
#                                                                        #
#  Thoughts for improvements: Have X and Y keys as input variables?      #
#                                                                        #
##########################################################################


import os, sys, time, shutil
from datetime import datetime
sys.path.append("/home/jchishol/TRecNet")
sys.path.append("home/jchishol/")
os.environ["CUDA_VISIBLE_DEVICES"]="1"    # These are the GPUs visible for training
from argparse import ArgumentParser
from contextlib import redirect_stdout

import numpy as np
import vector
import itertools
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import uproot
import json

import tensorflow as tf
import keras
from keras.layers import Conv1D, Flatten, Dense, Input, concatenate, Masking, LSTM, TimeDistributed, Lambda, Reshape, Multiply, BatchNormalization, Bidirectional
from keras import regularizers 
from keras import initializers
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
import keras.backend as K  
from keras.optimizers import *
import keras_tuner as kt
#from clr_callback import * 

from MLUtil import Utilities
import Scaler as Scaler
import ml.ShapeTimesteps as ShapeTimesteps
from ModelBuilder import TRecNet_Model, ModelBuilder

import tracemalloc
tracemalloc.start()


class Training:
    """
    A class for training and hypertuning neural networks.
    """  
    
    def __init__(self, config_file):
        """
        Initializes a training object.

            Parameters:
                config_file (dict): Training configuration dictionary.
                
            Attributes:
                train_file (str): Name (including path) of data file to train on (taken from config file).
                xmm_file (str): Name (including path) of x maxmean file to train on (taken from config file).
                ymm_file (str): Name (including path) of y maxmean file to train on (taken from config file).
                split (float): Percentage of data from <train_file> that will be given to training, while the remainder is used for validation (taken from config file).
                pretrain_file (str): Name (including path) of the jet pretrain model file.
                frozen_file (str): Name (including path) of the frozen model file.
                max_epochs (int): Maximum number of epochs to train on (taken from config file).
                max_trials (int): Maximum number of trials for Bayesian Optimization hypertuning (taken from config file).
                patience (int): Patience (max number of epochs with no improvements) to train with (taken from config file).
                X_keys (list of str): Keys for the (original scale) X variables.
                Y_keys (list of str): Keys for the (original scale) Y variables.
                X_scaled_keys (list of str): Keys for the (maxmean scale) X variables.
                Y_scaled_keys (list of str): Keys for the (maxmean scale) Y variables.
                training_time (datetime): Time it takes to train the model.
                training_history (keras.model.history.history): Training history for the model.
        """
        self.train_file = config_file["data"]
        self.xmm_file = config_file["xmaxmean"]
        self.ymm_file = config_file["ymaxmean"]
        self.split = config_file["split"][0]/(config_file["split"][0]+config_file["split"][1])   # Gave 85% to train file, now want 70% for the actual training ([0]=% in train, [1]=% in val, [2]=% in test)
        
        self.max_epochs = config_file["max_epochs"]
        self.patience = config_file["patience"]
        
        self.X_keys = None
        self.Y_keys = None
        self.X_scaled_keys = None
        self.Y_scaled_keys = None
        
        self.training_time = None
        self.training_history = None

        # Want to make sure we've got GPU
        if tf.config.list_physical_devices('GPU')==[]:
            print("WARNING: Networks will be trained on CPU. Move into container to use GPU.") 
          
          
    def set_create_params(self, create_config):
        """
        Sets the training parameters from the config file.

            Parameters:
                create_config (dict): Creation configuration dictionary.
        """
        
        self.jet_pretrain_file = create_config["jet_pretrain_model"]
        self.bb_pretrain_file = create_config["bb_pretrain_model"]
        
        
    def set_unfreeze_params(self, unfreeze_config):
        """
        Sets the training parameters from the config file.

            Parameters:
                unfreeze_config (dict): Unfreeze configuration dictionary.
        """
        
        self.frozen_file = unfreeze_config["frozen_model"]
        self.frozen_model_id = unfreeze_config["frozen_file"].split('/')[-1]
         
            
    def set_train_params(self, train_config):
        """
        Sets the training parameters from the config file.

            Parameters:
                train_config (dict): Training configuration dictionary.
        """
        
        self.initial_lr = train_config["initial_lr"]
        self.final_lr_div = train_config["final_lr_div"]
        self.lr_decay_step = train_config["lr_decay_step"]
        self.lr_power = train_config["lr_power"]
        self.batch_size = train_config["batch_size"]
        
        
    def set_hyper_config(self, hyper_config):
        """
        Sets the hypertuning configuration for later use.

            Parameters:
                hyper_config (dict): Hypertuning configuration dictionary.
        """
        
        self.tuner_type = hyper_config["tuner"]
        self.hyper_config = hyper_config["hyperparams"]
 
 
    def load_and_prep(self, Model):
        """
        Loads, pre-processes, and splits the data, such that it is ready to use for training.

            Parameters:
                Model (Model object): Model that we'll be training.

            Returns:
                trainX_jets (array): Pre-processed jets input training data.
                valX_jets (array): Pre-processed jets input validation data.
                trainX_other (array): Pre-processed other input training data.
                valX_other (array): Pre-processed other input validation data.
                trainY (array): Pre-processed output training data.
                valY (array): Pre-processed output validation data.
        """
        
        # Create an object to use utilities
        processor = Utilities()

        # These are the keys for what we're feeding into the pre-processing, and getting back in the end
        # X and Y variables to be used (NOTE: later have option to feed these in?)
        self.X_keys, self.Y_keys = processor.getInputKeys(Model.model_name, Model.n_jets)

        # Load maxmean
        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)

        # Pre-process the data
        totalX_jets, totalX_other, totalY, self.X_scaled_keys, self.Y_scaled_keys = processor.scale_and_shape(self.train_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, Model.n_jets, Model.mask_value)
        
        # Save shapes for later
        Model.had_shape = sum('th_' in key or 'wh_' in key for key in self.Y_scaled_keys)
        Model.lep_shape = sum('tl_' in key or 'wl_' in key for key in self.Y_scaled_keys)
        Model.ttbar_shape = sum('ttbar_' in key for key in self.Y_scaled_keys)
        Model.bbbar_shape = sum('b_' in key or 'bbar_' in key for key in self.Y_scaled_keys)

        # Split the data
        trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = train_test_split(totalX_jets, totalX_other, totalY, train_size=self.split)
        
        # Save the shapes for later
        Model.jets_shape = trainX_jets.shape
        Model.other_shape = trainX_other.shape
        
        return trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY

 
    def save_model(self, Model):
        """
        Saves the model itself, the training history, and plots of the training loss.

            Parameters:
                Model (Model object): Model to be saved.

            Returns:
                Saves model as <model_name>/<model_name>_%Y%m%d%H%M%S.keras, saves training history as <model_name>/<model_name>_%Y%m%d%H%M%S_TrainHistory.npy
                and saves loss plots in the <model_name> directory.
        """

        print('Saving model...')
        
        dir = 'trained_models/'+Model.model_v+'/'+Model.model_name+'/'+Model.model_id
        
        # Create directory for saving things in if it doesn't exist
        if not os.path.exists(dir): 
            os.makedirs(dir) 

        # Save model and history
        Model.model.save(dir+'/'+Model.model_id+'.keras')
        np.save(dir+'/'+Model.model_id+'_TrainHistory.npy',self.training_history)
        
        # Save maxmean scaling files
        shutil.copy(self.xmm_file, dir)
        shutil.copy(self.ymm_file, dir)

        # Save important information about this model into a text file
        file = open(dir+'/'+Model.model_id+"_Info.txt", "w")
        file.write("Model ID: %s \n" % Model.model_id)
        if Model.unfreeze: file.write("Frozen Model ID: %s \n" % self.frozen_model_id)
        if Model.use_JetPretraining: file.write("JetPretrain Model: %s \n" % self.jet_pretrain_file)
        if Model.use_bbPretraining: file.write("bbPretrain Model: %s \n" % self.bb_pretrain_file)
        file.write("\n ---------------------------------------------------  \n")
        file.write("Training Data File: %s \n" % self.train_file)
        file.write("X Maxmean File: %s \n" % self.xmm_file)
        file.write("Y Maxmean File: %s \n" % self.ymm_file)
        file.write("\n ---------------------------------------------------  \n")
        file.write("X Keys: "+', '.join(self.X_keys)+'\n')
        file.write("X Scaled Keys: "+', '.join(self.X_scaled_keys)+'\n')
        file.write("Y Keys: "+', '.join(self.Y_keys)+'\n')
        file.write("Y Scaled Keys: "+', '.join(self.Y_scaled_keys)+'\n')
        file.write("\n ---------------------------------------------------  \n")
        file.write("Learning Rate: Polynomial Decay with initial_lr=%s, final_lr=%s, decay_step=%s, and power=%s \n" % (self.initial_lr, (self.initial_lr/self.final_lr_div), self.lr_decay_step, self.lr_power))
        file.write("Batch Size: %s \n" % self.batch_size)
        file.write("Max Number of Epochs: %s \n" % self.max_epochs)
        file.write("Number of Epochs Used: %s \n" % len(self.training_history['loss']))
        file.write("Patience: %s \n" % self.patience)
        file.write("Training Time: %02d:%02d:%02d:%02d \n" % (self.training_time.days, self.training_time.seconds // 3600, self.training_time.seconds // 60 % 60, self.training_time.seconds % 60))
        file.write("Training History: \n %s \n" % pd.DataFrame(self.training_history).to_string(index=False))
        file.write("\n ---------------------------------------------------  \n")
        file.write("Model Architecture:\n")
        with redirect_stdout(file): Model.model.summary(expand_nested=True, show_trainable=True)
        file.close()

        # Save training history plots
        if 'JetPretrainer' in Model.model_v or 'bbPretrainer' in Model.model_v:
            plt.figure(figsize=(9,6))
            plt.plot(self.training_history['loss'], label='training')
            plt.plot(self.training_history['val_loss'], label='validation')
            plt.xlabel('Epoch')
            plt.ylabel('Binary Cross Entropy Loss')
            plt.legend()
            plt.title(Model.model_name+' Binary Cross Entropy Loss')
            plt.savefig(dir+'/'+Model.model_id+'_BinaryCrossEntropy.png',bbox_inches='tight')

            plt.figure(figsize=(9,6))
            plt.plot(self.training_history['mae'], label='training')
            plt.plot(self.training_history['val_mae'], label='validation')
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend()
            plt.title(Model.model_name+' MAE Loss')
            plt.savefig(dir+'/'+Model.model_id+'_MAE.png',bbox_inches='tight')
        else:
            plt.figure(figsize=(9,6))
            plt.plot(self.training_history['loss'], label='training')
            plt.plot(self.training_history['val_loss'], label='validation')
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend()
            plt.title(Model.model_name+' MAE Loss')
            plt.savefig(dir+'/'+Model.model_id+'_MAE.png',bbox_inches='tight')
            
        plt.figure(figsize=(9,6))
        plt.plot(self.training_history['mse'], label='training')
        plt.plot(self.training_history['val_mse'], label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title(Model.model_name+' MSE Loss')
        plt.savefig(dir+'/'+Model.model_id+'_MSE.png',bbox_inches='tight')
        
        
    def train(self, Model):
        """
        Builds, trains, and saves the model.

            Parameters:
                Model (Model object): Model to be built and trained.
        """

        
        # Get the data
        trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = self.load_and_prep(Model)
        
        # Build the model
        modelbuilder = ModelBuilder(Model)
        if Model.use_JetPretraining and Model.use_bbPretraining:
            jet_pretrain_model = keras.layers.TFSMLayer(self.jet_pretrain_file, call_endpoints="serving_default")
            bb_pretrain_model = keras.layers.TFSMLayer(self.bb_pretrain_file, call_endpoints="serving_default")
            Model.model = modelbuilder.create_model(self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step, jet_pretrain_model=jet_pretrain_model, bb_pretrain_model=bb_pretrain_model)
        elif Model.use_JetPretraining:
            pretrain_model = keras.layers.TFSMLayer(self.jet_pretrain_file, call_endpoints="serving_default")
            Model.model = modelbuilder.create_model(self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step, jet_pretrain_model=pretrain_model)
        elif Model.use_bbPretraining:
            pretrain_model = keras.layers.TFSMLayer(self.jet_pretrain_file, call_endpoints="serving_default")
            Model.model = modelbuilder.create_model(self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step, jet_pretrain_model=pretrain_model)
        elif Model.unfreeze:
            Model.model = modelbuilder.create_model(self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step, frozen_file=self.frozen_file)
        else:
            Model.model = modelbuilder.create_model(self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step)    
        print(Model.model_id+' model has been built and compiled.')
            
        
        # Set early stopping (so no overfitting) and tensorboard callback (for monitoring)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir= "tensorboard_logs/fit/"+Model.model_id, histogram_freq=1)
        
        # Fit/train the model      
        start = datetime.now()
        history = Model.model.fit([trainX_jets, trainX_other], trainY, verbose=1, epochs=self.max_epochs, validation_data=([valX_jets, valX_other], valY), shuffle=True, callbacks=[early_stop, tensorboard_callback], batch_size=self.batch_size)
        end = datetime.now()
        self.training_time = end - start
        self.training_history = history.history
        
        # Save the model, its history, and loss plots
        self.save_model(Model)
        

    def save_hypertune_results(self,Model,tuner):
        """
        Save the hypertuning results.

            Parameters:
                Model (Model object): Model that we've been hypertuning.
                tuner (Keras tuner): Tuner that we've been hypertuning with.
        """
        
        print('Saving hyperparamter tuning results ...')
        
        dir = 'trained_models/'+Model.model_v+'/'+Model.model_name+'/hypertuning/'+Model.model_id+'_Hypertuning'
        
        # Create directory for saving things in if it doesn't exist
        if not os.path.exists(dir): 
            os.makedirs(dir) 

        
        # Save important information about this model into a text file
        file = open(dir+'/Hypertuning_Info.txt', "w")
        file.write("Model ID: %s \n" % Model.model_id)
        if Model.unfreeze: file.write("Frozen Model ID: %s \n" % self.frozen_model_id)
        if Model.use_JetPretrain: file.write("JetPretrain Model: %s \n" % self.jet_pretrain_file)
        if Model.use_bbPretrain: file.write("bbPretrain Model: %s \n" % self.bb_pretrain_file)
        file.write("Tuner: %s \n" % self.tuner_type)
        file.write("--------------------------------------------------- \n")
        file.write("Training Data File: %s \n" % self.train_file)
        file.write("X Maxmean File: %s \n" % self.xmm_file)
        file.write("Y Maxmean File: %s \n" % self.ymm_file)
        file.write("--------------------------------------------------- \n")
        file.write("X Keys: "+', '.join(self.X_keys)+'\n')
        file.write("X Scaled Keys: "+', '.join(self.X_scaled_keys)+'\n')
        file.write("Y Keys: "+', '.join(self.Y_keys)+'\n')
        file.write("Y Scaled Keys: "+', '.join(self.Y_scaled_keys)+'\n')
        file.write("--------------------------------------------------- \n")
        with redirect_stdout(file): tuner.search_space_summary(extended=True)
        file.write("--------------------------------------------------- \n")
        file.write("Total Hypertuning Time: %02d:%02d:%02d:%02d \n" % (self.training_time.days, self.training_time.seconds // 3600, self.training_time.seconds // 60 % 60, self.training_time.seconds % 60))
        with redirect_stdout(file): tuner.results_summary(10)
        file.write("--------------------------------------------------- \n")
        best_hps = tuner.get_best_hyperparameters()[0]
        #tuner.get_best_hyperparameters(num_trials=1).values
        file.write("Best initial_lr=%s: \n" % best_hps.get('initial_lr'))
        file.write("Best final_lr_div=%s: \n" % best_hps.get('final_lr_div'))
        file.write("Best lr_power=%s: \n" % best_hps.get('lr_power'))
        file.write("Best lr_decay_step=%s:\n " % best_hps.get('lr_decay_step'))
        file.write("Best batch_size=%s: \n" % best_hps.get('batch_size'))
        file.close()
        
        # Save ten best hyperparameter trials to a pandas dataframe?
        ten_best_hps = tuner.get_best_hyperparameters(num_trials=20)
        HP_list = []
        for hp in ten_best_hps:
            HP_list.append(hp.get_config()["values"])
        HP_df = pd.DataFrame(HP_list)
        #HP_df.to_csv(dir+"/top_ten_hp1.csv", index=False, na_rep='NaN')
        
        # Still need to see how this one does
        trials = tuner.oracle.get_best_trials(num_trials=20)
        HP_list = []
        for trial in trials:
            HP_list.append(trial.score)
        HP_df["Score"] = HP_list
        HP_df.to_csv(dir+"/top_hp.csv", index=False, na_rep='NaN')
        
        
    def hypertune(self, Model):
        """
        Hypertunes the model.

        Args:
            Model (Model object): Model we'll be hypertuning.
        """
    
            
        # Get the data
        trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = self.load_and_prep(Model)

        # Create hypertuner model
        hyper_model = self.TRecNetHyperModel(self, Model)
        if self.tuner_type=="Hyperband":
            print("Using Hyperband ...")
            tuner = kt.Hyperband(hyper_model, objective="val_loss", max_epochs = self.max_epochs, factor = 3, hyperband_iterations = 3, directory=ht_dir, project_name='hyperband_trials')
        else:
            print("Using BayesianOptimization ...")
            tuner = kt.BayesianOptimization(hyper_model, objective="val_loss", num_initial_points=5, max_trials = 20, directory=ht_dir, project_name='hyperband_trials')

        # Use callbacks
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir="tensorboard_logs/hypertuning/"+Model.model_id, histogram_freq=1)
        #csv_logger = CSVLogger(ht_dir+'/'+Model.model_id+'_Hypertuning/hypertuning_log.log',separator='\n')     # This didn't work great :/
        
        # Perform the search
        print('Hypertuning commencing...')
        start = datetime.now()
        tuner.search(x=[trainX_jets, trainX_other], y=trainY, validation_data=([valX_jets, valX_other], valY), epochs=self.max_epochs, callbacks=[early_stop, tensorboard_callback], shuffle=True,verbose=0)
        end = datetime.now()
        self.training_time = end - start

        # Save results
        self.save_hypertune_results(Model,tuner)
     

    class TRecNetHyperModel(kt.HyperModel):
        """
        A class based on the keras HyperModel.

            Parameters:
                kt (kt.HyperModel): Hypermodel.
        """

        def __init__(self, trainer, Model):
            """
            Initializes the hyper model.

                Parameters:
                    trainer (Training object): The training object that is calling this object.
                    Model (Model): Model we're hypertuning.
            """
            self.trainer = trainer  # Need this so we can inherit its functions, attributes, etc.
            self.Model = Model
            
            
        def get_hyperparams(self, hp):
            """
            Read the hyperparameter space from the configuration file and make a dictionary of hyperparameters.

                Parameters:
                    hp (?): Hyperparameter argument.

                Returns:
                    hp_dic (dict): Dictionary of hyperparameters.
            """
            
            hp_dic = {}
            for hyperparam, specs in self.trainer.hyper_config.items():
                if specs["type"]=="choice":
                    hp_dic[hyperparam] = hp.Choice(name=hyperparam, values = specs["choices"])
                elif specs["type"]=="int":
                    hp_dic[hyperparam] = hp.Int(name=hyperparam, min_value = specs["min_value"], max_value = specs["max_value"], step = specs["step"], sampling = specs["sampling"])
                elif specs["type"]=="float":
                    hp_dic[hyperparam] = hp.Float(name=hyperparam, min_value = specs["min_value"], max_value = specs["max_value"], step = specs["step"], sampling = specs["sampling"])
                else:
                    print("Can currently only handle choice, int, and float hyperparam types.")
                    sys.exit()
            return hp_dic
        
        
        def fit(self, hp, model, *args, **kwargs):
            """
            Fit the model.

                Parameters:
                    hp (?): Hyperparameter argument.
                    model (Model): Model to train.

                Returns:
                    (model object): Fitted model.
            """
            
            # Need this function to use batch_size as a hyperparameter.
            return model.fit(*args, batch_size=self.get_hyperparams(hp)["batch_size"],**kwargs,)
            
            
        def build(self, hp):
            """
            Build the hyperparameter model.

                Parameters:
                    hp (?): Hyperparameter argument.

                Returns:
                    model (model object): Model built with hyperparameters.
            """

            # Defining a set of hyperparametrs for tuning and a range of values for each
            hp_dic = self.get_hyperparams(hp)
            
            # Build the model
            modelbuilder = ModelBuilder(self.Model)
            if self.Model.use_JetPretraining and self.Model.use_bbPretraining:
                jet_pretrain_model = keras.layers.TFSMLayer(self.trainer.jet_pretrain_file, call_endpoints="serving_default")
                bb_pretrain_model = keras.layers.TFSMLayer(self.trainer.bb_pretrain_file, call_endpoints="serving_default")
                self.Model.model = modelbuilder.create_model(hp_dic['initial_lr'], hp_dic['final_lr_div'], hp_dic['lr_power'], hp_dic['lr_decay_step'], jet_pretrain_model=jet_pretrain_model, bb_pretrain_model=bb_pretrain_model)
            elif self.Model.use_JetPretraining:
                pretrain_model = keras.layers.TFSMLayer(self.trainer.jet_pretrain_file, call_endpoints="serving_default")
                self.Model.model = modelbuilder.create_model(hp_dic['initial_lr'], hp_dic['final_lr_div'], hp_dic['lr_power'], hp_dic['lr_decay_step'], jet_pretrain_model=pretrain_model)
            elif self.Model.use_bbPretraining:
                pretrain_model = keras.layers.TFSMLayer(self.trainer.jet_pretrain_file, call_endpoints="serving_default")
                self.Model.model = modelbuilder.create_model(hp_dic['initial_lr'], hp_dic['final_lr_div'], hp_dic['lr_power'], hp_dic['lr_decay_step'], bb_pretrain_model=pretrain_model)
            elif self.Model.unfreeze:
                self.Model.model = modelbuilder.create_model(hp_dic['initial_lr'], hp_dic['final_lr_div'], hp_dic['lr_power'], hp_dic['lr_decay_step'], frozen_file=self.trainer.frozen_file)
            else:
                self.Model.model = modelbuilder.create_model(hp_dic['initial_lr'], hp_dic['final_lr_div'], hp_dic['lr_power'], hp_dic['lr_decay_step'])    
            print(self.Model.model_id+' model has been built and compiled.')

        
            return self.Model.model