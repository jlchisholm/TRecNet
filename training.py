##########################################################################
#                                                                        #
#  training.py                                                           #
#  Author: Jenna Chisholm                                                #
#  Updated: June.4/24                                                    #
#                                                                        #
#  Defines classes and functions relevant for training and hypertuning   #
#  neural networks.                                                      # 
#                                                                        #
#  Thoughts for improvements: Have X and Y keys as input variables?      #
#                                                                        #
##########################################################################


import os, sys, time
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

from MLUtil import *
import normalize_new
import shape_timesteps_new

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
                jets_shape (tuple): Shape of the jets input.
                other_shape (tuple): Shape of the other input.
        """
        self.train_file = config_file["data"]
        self.xmm_file = config_file["xmaxmean"]
        self.ymm_file = config_file["ymaxmean"]
        self.split = config_file["split"][0]/(config_file["split"][0]+config_file["split"][1])   # Gave 85% to train file, now want 70% for the actual training ([0]=% in train, [1]=% in val, [2]=% in test)
        self.pretrain_file = config_file["jet_pretrain"]
        self.frozen_file = config_file["frozen_model"]
        self.max_epochs = config_file["max_epochs"]
        self.patience = config_file["patience"]
        self.X_keys = None
        self.Y_keys = None
        self.X_scaled_keys = None
        self.Y_scaled_keys = None
        self.jets_shape = None
        self.other_shape = None

        # Want to make sure we've got GPU
        if tf.config.list_physical_devices('GPU')==[]:
            print("WARNING: Networks will be trained on CPU. Move into container to use GPU.") 
            
            
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
        
        
    def set_hyper_config(self, tuner_type, hyper_config):
        """
        Sets the hypertuning configuration for later use.

            Parameters:
                tuner_type (str): Tuning algorithm to be used (e.g. 'Hyperband' or 'Bayesian Optimization').
                hyper_config (dict): Hypertuning configuration dictionary.
        """
        
        self.tuner_type = tuner_type
        self.hyper_config = hyper_config
 
 
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
        
        #current, peak = tracemalloc.get_traced_memory()
        #print(f"Current memory usage is {current / 10**3}KB; Peak was {peak / 10**3}KB; Diff = {(peak - current) / 10**3}KB")


        # Pre-process the data
        totalX_jets, totalX_other, totalY, self.X_scaled_keys, self.Y_scaled_keys = processor.prepData(self.train_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, Model.n_jets, Model.mask_value)

        #current, peak = tracemalloc.get_traced_memory()
        #print(f"Current memory usage is {current / 10**3}KB; Peak was {peak / 10**3}KB; Diff = {(peak - current) / 10**3}KB")


        # Split the data
        trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = train_test_split(totalX_jets, totalX_other, totalY, train_size=self.split)
        
        # Save the shapes for later
        self.jets_shape = trainX_jets.shape
        self.other_shape = trainX_other.shape
        
        return trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY
    
    
    def build_model(self, model_name, mask_value, initial_lr, final_lr_div, lr_power, lr_decay_step, pretrain_model=None):
        """
        Build the model architecture.

            Parameters:
                model_name (str): Name of the model.
                mask_value (int): Value that will mask non-existent jets.
                initial_lr (float): Initial learning rate.
                final_lr_div (float): Final learning rate divisor (i.e. final_lr = initial_lr/final_lr_div).
                lr_power (float): Power for the decaying polynomial learning rate.
                lr_decay_step (int): Learning rate decay step.
            
            Optional:
                pretrain_model (Model object): Jet pre-trained model (default: None).

            Returns:
                model (Model object): Built model.
        """
        
        # Print before
        #current, peak = tracemalloc.get_traced_memory()
        #print(f"Current (before) memory usage is {current / 10**3}KB; Peak was {peak / 10**3}KB; Diff = {(peak - current) / 10**3}KB")
        
        # Might need to clear session so keras doesn't run out of memory
        K.clear_session()
        
        # Print after
        #current, peak = tracemalloc.get_traced_memory()
        #print(f"Current (after) memory usage is {current / 10**3}KB; Peak was {peak / 10**3}KB; Diff = {(peak - current) / 10**3}KB")
         
        print('Building model...')

        # This will help for dimension of output layers
        had_shape = sum('th_' in key or 'wh_' in key for key in self.Y_scaled_keys)
        lep_shape = sum('tl_' in key or 'wl_' in key for key in self.Y_scaled_keys)
        ttbar_shape = sum('ttbar_' in key for key in self.Y_scaled_keys)

        # Input layers
        jet_input = Input(shape=(self.jets_shape[1], self.jets_shape[2]),name='jet_input')
        other_input = Input(shape=(self.other_shape[1],),name='other_input')

        # Mask the jets that are just there for padding (do I want this to be what's put into the flattened jets? or no, since it's being reshaped?) and use some TDDense layers
        Mask = Masking(mask_value, name='masking_jets')(jet_input)
        Maskshape = Reshape((self.jets_shape[1], self.jets_shape[2]), name='reshape_masked_jets')(Mask)
        TDDense11 = TimeDistributed(Dense(128, activation='relu'), name='TDDense128')(Maskshape)
        TDDense12 = TimeDistributed(Dense(64, activation='relu'), name='TDDense64')(TDDense11)

        # Concatenate flattened jets and other and use some dense layers (but not for TRecNet+ttbar+JetPretrain, since this is done in JetPretrainer)
        if model_name!='TRecNet+ttbar+JetPretrain':
            flat_jets =  Flatten(name ='flattened_jets')(jet_input) 
            concat0 = concatenate([other_input, flat_jets], name = 'concat_jets_other')
            PreDense1 = Dense(256, activation='relu', name = 'dense256_1')(concat0)
            PreDense2 = Dense(256, activation='relu', name = 'dense256_2')(PreDense1) 

        # This is mostly all we need for the rest of JetPretrainer
        if model_name=='JetPretrainer':

            # Final output
            output = Dense(self.jets_shape[1], activation='sigmoid', name='jet_match_output')(PreDense2)
            
        # Lots more layers for the actual training
        else:

            # Use sigmoid to give jets a weight
            if model_name=='TRecNet+ttbar+JetPretrain':
                pretrain_model.trainable = False                                      # Freezing the jet pretrain model (i.e. want to use the previously trained weights)
                pretrain = pretrain_model([jet_input,other_input], training=False)    # Putting the inputs into the pretrain model
                Shape_Dot = Reshape((-1,1), name='reshape')(pretrain)
            else:
                PreDense3 = Dense(self.jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(PreDense2)
                Shape_Dot = Reshape((-1,1), name='reshape')(PreDense3)
            Dot_jets = Multiply(name='weight_jets')([Shape_Dot, TDDense12])

            # Use some more TDDense layers with the weighted jets
            TDDense13 = TimeDistributed(Dense(256, activation='relu'), name='TDDense256_1')(Dot_jets)
            TDDense14= TimeDistributed(Dense(256, activation='relu'), name='TDDense256_2')(TDDense13)
            flat_right = Flatten(name='flattened_weighted_jets')(TDDense14)
        
            # On the other side, do some dense layers
            Dense21 = Dense(128, activation='relu', name='dense128')(other_input)
            Dense22 = Dense(64, activation='relu', name='dense64')(Dense21)
            flat_other = Flatten(name='flattened_other')(Dense22)
        
            # Concatenate the two sides
            concat = concatenate([flat_other, flat_right], name = 'concat_everything')
        
            # Output leptonic side
            ldense1 = Dense(256, activation='relu', name='ldense256')(concat)
            ldense2 = Dense(128, activation='relu', name='ldense128')(ldense1)
            loutput = Dense(lep_shape, name='lep_output')(ldense2)
            
            # Output hadronic(+ttbar) side
            hconcat = concatenate([loutput, concat])
            hdense1 = Dense(256, activation='relu', name='hdense256')(hconcat)
            hdense2 = Dense(128, activation='relu', name='hdense128')(hdense1)
            houtput = Dense(had_shape+ttbar_shape, name='had_output')(hdense2)
            
            # Final output
            output = concatenate([houtput, loutput], name='output')

        # Putting the model together
        model = keras.models.Model(inputs=[jet_input, other_input], outputs=output)
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr, decay_steps=lr_decay_step,end_learning_rate=initial_lr/final_lr_div,power=lr_power)
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        if model_name == 'JetPretrainer':
            model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['mae','mse'],jit_compile=False)
        else:
            model.compile(loss='mae', optimizer= optimizer, metrics=['mse'],jit_compile=False)

        return model 
   
 
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
        
        dir = Model.model_name+'/'+Model.model_id
        
        # Create directory for saving things in if it doesn't exist
        if not os.path.exists(Model.model_name):
            os.mkdir(Model.model_name)
        if not os.path.exists(dir):
            os.mkdir(dir)

        # Save model and history
        Model.model.save(dir+'/'+Model.model_id+'.keras')
        np.save(dir+'/'+Model.model_id+'_TrainHistory.npy',Model.training_history)

        # Save important information about this model into a text file
        file = open(dir+'/'+Model.model_id+"_Info.txt", "w")
        file.write("Model Name: %s \n" % Model.model_name)
        file.write("Model ID: %s \n" % Model.model_id)
        if 'Unfrozen' in Model.model_name: file.write("Frozen Model ID: %s \n" % Model.frozen_model_id)
        if 'TRecNet+ttbar+JetPretrain' in Model.model_name: file.write("JetPretrain Model: %s \n" % self.pretrain_file)
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
        file.write("Number of Epochs Used: %s \n" % len(Model.training_history['loss']))
        file.write("Patience: %s \n" % self.patience)
        #file.write("Training Time: %s \n" % time.strftime("%H:%M:%S", time.gmtime(Model.training_time)))
        file.write("Training Time: %02d:%02d:%02d:%02d \n" % (Model.training_time.days, Model.training_time.seconds // 3600, Model.training_time.seconds // 60 % 60, Model.training_time.seconds % 60))
        file.write("Training History: \n %s \n" % pd.DataFrame(Model.training_history).to_string(index=False))
        file.write("\n ---------------------------------------------------  \n")
        file.write("Model Architecture:\n")
        with redirect_stdout(file): Model.model.summary(expand_nested=True, show_trainable=True)
        file.close()

        # Save training history plots
        if Model.model_name=='JetPretrainer':
            plt.figure(figsize=(9,6))
            plt.plot(Model.training_history['loss'], label='training')
            plt.plot(Model.training_history['val_loss'], label='validation')
            plt.xlabel('Epoch')
            plt.ylabel('Binary Cross Entropy Loss')
            plt.legend()
            plt.title(Model.model_name+' Binary Cross Entropy Loss')
            plt.savefig(dir+'/'+Model.model_id+'_BinaryCrossEntropy',bbox_inches='tight')

            plt.figure(figsize=(9,6))
            plt.plot(Model.training_history['mae'], label='training')
            plt.plot(Model.training_history['val_mae'], label='validation')
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend()
            plt.title(Model.model_name+' MAE Loss')
            plt.savefig(dir+'/'+Model.model_id+'_MAE',bbox_inches='tight')
        else:
            plt.figure(figsize=(9,6))
            plt.plot(Model.training_history['loss'], label='training')
            plt.plot(Model.training_history['val_loss'], label='validation')
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend()
            plt.title(Model.model_name+' MAE Loss')
            plt.savefig(dir+'/'+Model.model_id+'_MAE',bbox_inches='tight')
            
        plt.figure(figsize=(9,6))
        plt.plot(Model.training_history['mse'], label='training')
        plt.plot(Model.training_history['val_mse'], label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title(Model.model_name+' MSE Loss')
        plt.savefig(dir+'/'+Model.model_id+'_MSE',bbox_inches='tight')
        
        
    def train(self, Model):
        """
        Builds, trains, and saves the model.

            Parameters:
                Model (Model object): Model to be built and trained.
        """

        
        # Get the data
        trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = self.load_and_prep(Model)
        
        # If doing the model with jet pretraining, we want to unfreeze and fine tune the weights
        if Model.model_name=='TRecNet+ttbar+JetPretrainUnfrozen':
            
            # Load the frozen model to start with
            Model.model = keras.layers.TFSMLayer(self.frozen_file, call_endpoint="serving_default")
            
            # Find the jet pre-training layer and unfreeze all those sublayers
            for layer in Model.model.layers:
                if isinstance(layer, keras.Model):
                    layer.trainable = True 
            
            # Use a smaller learning rate since we're fine-tuning now
            lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.initial_lr, decay_steps=self.lr_decay_step,end_learning_rate=self.initial_lr/self.final_lr_div,power=self.lr_power)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            
            # Recompile the model    
            Model.model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])
            
        else:
        
            # Build the model
            # Need to load jet pretraining if TRecNet+ttbar+JetPretrain
            if Model.model_name=='TRecNet+ttbar+JetPretrain':
                pretrain_model = keras.layers.TFSMLayer(self.pretrain_file, call_endpoints="serving_default")
                Model.model = self.build_model(Model.model_name, Model.mask_value, self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step, pretrain_model=pretrain_model)
            else:
                Model.model = self.build_model(Model.model_name, Model.mask_value, self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step)
            print(Model.model_name+' model has been built and compiled.')
        
        
        # Set early stopping (so no overfitting) and tensorboard callback (for monitoring)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir= "tensorboard_logs/fit/"+Model.model_id, histogram_freq=1)
        
        # Fit/train the model      
        start = datetime.now()
        history = Model.model.fit([trainX_jets, trainX_other], trainY, verbose=1, epochs=self.max_epochs, validation_data=([valX_jets, valX_other], valY), shuffle=True, callbacks=[early_stop, tensorboard_callback], batch_size=self.batch_size)
        end = datetime.now()
        Model.training_time = end - start
        Model.training_history = history.history
        
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
        
        dir = Model.model_name+'/hypertuning/'+Model.model_id+'_Hypertuning'
        
        # Save important information about this model into a text file
        file = open(dir+'/Hypertuning_Info.txt', "w")
        file.write("Model Name: %s \n" % Model.model_name)
        file.write("Model ID: %s \n" % Model.model_id)
        if 'Unfrozen' in Model.model_name: file.write("Frozen Model ID: %s \n" % Model.frozen_model_id)
        if 'TRecNet+ttbar+JetPretrain' in Model.model_name: file.write("JetPretrain Model: %s \n" % self.pretrain_file)
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
        #file.write("Total Hypertuning Time: %s \n" % time.strftime("%H:%M:%S", time.gmtime(Model.training_time)))
        file.write("Total Hypertuning Time: %02d:%02d:%02d:%02d \n" % (Model.training_time.days, Model.training_time.seconds // 3600, Model.training_time.seconds // 60 % 60, Model.training_time.seconds % 60))
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
    
        # Create directory for results
        ht_dir = Model.model_name+'/hypertuning/'+Model.model_id+'_Hypertuning'
        if not os.path.exists(ht_dir):
            os.mkdir(ht_dir)
            
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
        Model.training_time = end - start

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
            
            # If doing the model with jet pretraining, we want to unfreeze and fine tune the weights
            if self.Model.model_name=='TRecNet+ttbar+JetPretrainUnfrozen':
                
                # Load the frozen model to start with
                self.Model.model = keras.layers.TFSMLayer(self.trainer.frozen_file, call_endpoint="serving_default")
                
                # Double check that we're unfreezing the correct layer
                if self.Model.model.layers[4].trainable:
                    print('Trying to fine-tune the wrong layer. Look at model summary below and find the correct index for the jet pretrained layer.')
                    print(self.Model.model.summary(expand_nested=True, show_trainable=True))
                    sys.exit()

                # Unfreeze ALL jet pretrained layers
                self.Model.model.layers[4].trainable = True    
                
                # Use a smaller learning rate since we're fine-tuning now
                lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=hp_dic['initial_lr'], decay_steps=hp_dic['lr_decay_step'],end_learning_rate=hp_dic['initial_lr']/hp_dic['final_lr_div'],power=hp_dic['lr_power'])
                optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
                
                # Recompile the model    
                self.Model.model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])
            
            else:
            
                # Need to load jet pretraining if TRecNet+ttbar+JetPretrain
                if self.Model.model_name=='TRecNet+ttbar+JetPretrain':
                    pretrain_model = keras.models.load_model(self.trainer.pretrain_file)
                    self.Model.model = self.trainer.build_model(self.Model.model_name, self.Model.mask_value, hp_dic['initial_lr'], hp_dic['final_lr_div'], hp_dic['lr_power'], hp_dic['lr_decay_step'], pretrain_model = pretrain_model)
                else:
                    self.Model.model = self.trainer.build_model(self.Model.model_name, self.Model.mask_value, hp_dic['initial_lr'], hp_dic['final_lr_div'], hp_dic['lr_power'], hp_dic['lr_decay_step'])
        
            return self.Model.model
            
        


# ------ MAIN CODE ------ #

# Set up argument parser
parser = ArgumentParser()
parser.add_argument('--model_name', help='Name of the model to be trained.', type=str, required=True, choices=['TRecNet','TRecNet+ttbar','JetPretrainer','TRecNet+ttbar+JetPretrain','TRecNet+ttbar+JetPretrainUnfrozen'])
parser.add_argument('--config_file', help="File (including path) with training (or hypertuning) specifications.", type=str, required=True)
parser.add_argument('--hypertune', help="Use this flag to hypertune your model.", action="store_true")

# Only need tuner if we're hypertuning
args, tuner_arg = parser.parse_known_args()
if args.hypertune:
    parser.add_argument('--tuner', required=True, choices=["Hyperband","BayesianOptimization"])
    #parser.parse_args(tuner_arg, namespace = args)

# Get config file
config = json.load(open(args.config_file))


# Check that there is a pre-train model, with the same number of jets, if needed
if '+JetPretrain' in args.model_name:
    if config["jet_pretrain"]==None:
        print('Please provide a jet pretrained model in order to train TRecNet+ttbar+JetPretrain.')
        sys.exit()
    else:
        pretrain_n_jets = int(config["jet_pretrain"].split('/')[-1].split('jets')[0].split('_')[-1])
        if pretrain_n_jets != config["njets"]:
            print("Please provide a jet pretrain model with the same number of jets as you desire.")
            sys.exit()

# Check that there is a frozen version of the model, with the same number of jets, if needed
if 'Unfrozen' in args.model_name:
    if config["frozen_model"]==None:
        print('Please provide a frozen version of the model in order to unfreeze and fine-tune weights.')
        sys.exit()
    else:
        frozen_n_jets = int(config["frozen_model"].split('/')[-1].split('jets')[0].split('_')[-1])
        if frozen_n_jets != config["njets"]:
            print("Please provide a frozen model with the same number of jets as you desire.")
            sys.exit()

# Instantiate model
Model = TRecNet_Model(args.model_name, n_jets=config["njets"])

# Hypertune
if args.hypertune:
    print('Beginning hypertuning for '+args.model_name+'...')
    hyper_config = config["hypertuning"]
    Trainer = Training(config)
    Trainer.set_hyper_config(tuner_arg[1],config["hypertuning"])
    Trainer.hypertune(Model)

# Train
else:
    print('Beginning training for '+args.model_name+'...')
    if not os.path.exists(Model.model_name+'/'+Model.model_id):
        os.mkdir(Model.model_name+'/'+Model.model_id)
    Trainer = Training(config)
    Trainer.set_train_params(config["training"])
    Trainer.train(Model)


print('done :)')






        
        
