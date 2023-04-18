######################################################################
#                                                                    #
#  machine_learning.py                                               #
#  Author: Jenna Chisholm                                            #
#  Updated: Apr.5/23                                                 #
#                                                                    #
#  Defines classes and functions relevant for training and testing   #
#  neural networks.                                                  # 
#                                                                    #
#  Thoughts for improvements: Have X and Y keys as input variables?, #
#  include validation module, make sure everything actually works.   #
#                                                                    #
######################################################################


import os, sys, time
sys.path.append("/home/jchishol/TRecNet")
sys.path.append("home/jchishol/")
os.environ["CUDA_VISIBLE_DEVICES"]="0"    # These are the GPUs visible for training
from argparse import ArgumentParser

import numpy as np
import vector
import itertools
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import uproot

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Flatten, Dense, Input, concatenate, Masking, LSTM, TimeDistributed, Lambda, Reshape, Multiply, BatchNormalization, Bidirectional
from keras import regularizers 
from keras import initializers
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import keras.backend as K  
from keras.optimizers import *
#from clr_callback import * 

import normalize_new
import shape_timesteps_new





class TRecNet_Model:
    """
    A class for creating a machine learning model object.
    """

    def __init__(self, model_name, model_id=None, n_jets=None):
        """
        Initializes a machine learning model object.

            Parameters:
                model_name (str): Name of the model (e.g. 'TRecNet+ttbar').
                model_id (str): Unique model identifier (default: None).
                n_jets (int): Number of jets the model is trained with (default: None).

            Attributes:
                model_name (str): Name of the model (e.g. 'TRecNet+ttbar').
                model_id (str): Unique model identifier, based on model name, number of jets, and time it was created.
                n_jets (int): Number of jets the model is trained with.
                model (keras.model): Trained keras model.
                training_history (keras.model.history.history): Training history for the model.
        """
        
        self.model_name = model_name
        self.model_id = time.strftime(model_name+"_"+str(n_jets)+"jets_%Y%m%d_%H%M%S") if model_id==None else model_id   # Model unique save name (based on the date)
        self.n_jets = n_jets if model_id==None else int(model_id.split('_')[1].split('jets')[0]) # If not given, get from model_id
        self.mask_value = -2   # Define here so it's consist between model building and jet timestep building
        self.model = None
        self.frozen_model_id = None
        self.training_history = None


class Utilities:
    """
    A class containing useful functions for training, validating, and testing.
    
        Methods:
            getInputKeys: Gets lists of the (original scale) X and Y variable keys.
            loadMaxMean: Loads the max-mean dictionaries.
            scale: Scale the X and Y data such that they have a mean of 0, range of (-1,1), and encode phi variables in px and py (or cosine and sine).
            unscale: Unscale the X and Y data back to the original scale and variables.
            prepData: Prepares data for training by performing a mean-max scaling and phi-encoding, and then splitting dataset into (time-stepped) jets, other, and y data.
    """

    def __init__(self):
        pass

    def getInputKeys(self, model_name, n_jets):
        """
        Gets lists of the (original scale) X and Y variable keys.

            Parameters:
                model_name (str): Name of the model (e.g. 'TRecNet+ttbar').
                n_jets (int): Number of jets the model is or will be trained on.
                
            Returns:
                X_keys (list of str): Keys for the (original scale) X variables.
                Y_keys (list of str): Keys for the (original scale) Y variables.
        """

        X_keys = ['j'+str(i+1)+'_'+v for i, v in itertools.product(range(n_jets),['pt','eta','phi','m','isbtag'])] + ['lep_pt', 'lep_eta', 'lep_phi', 'met_met', 'met_phi']
        Y_keys = ['th_pt', 'th_eta','th_phi','th_m', 'wh_pt', 'wh_eta', 'wh_phi', 'wh_m', 'tl_pt', 'tl_eta', 'tl_phi', 'tl_m', 'wl_pt', 'wl_eta', 'wl_phi', 'wl_m']
        if 'ttbar' in model_name:
            Y_keys.extend(['ttbar_pt','ttbar_eta','ttbar_phi','ttbar_m'])
        if model_name=='JetPretrainer': 
            Y_keys = ['j'+str(i+1)+'_isTruth' for i in range(n_jets)]

        return X_keys, Y_keys


    def loadMaxMean(self, xmm_file, ymm_file):
        """
        Loads the max-mean dictionaries.
        
            Parameters:
                xmm_file (str): X maxmean file name (including path).
                ymm_file (str): Y maxmean file name (including path).

            Returns:
                X_maxmean_dic (dict): Dictionary of max and mean for X variables.
                Y_maxmean_dic (dict): Dictionary of max and mean for Y variables.
        """

        X_maxmean_dic = np.load(xmm_file,allow_pickle=True).item()
        Y_maxmean_dic = np.load(ymm_file,allow_pickle=True).item()

        return X_maxmean_dic, Y_maxmean_dic


    def scale(self, dataset, X_keys, Y_keys, X_maxmean_dic, Y_maxmean_dic):
        """
        Scale the X and Y data such that they have a mean of 0, range of (-1,1), and encode phi variables in px and py (or cosine and sine).

            Parameters:
                dataset (h5py dataset): Training data.
                X_keys (list of str): Keys for the (original scale) X variables.
                Y_keys (list of str): Keys for the (original scale) Y variables.
                X_maxmean_dic (dict): Dictionary of max and mean for X variables.
                Y_maxmean_dic (dict): Dictionary of max and mean for Y variables.

            Returns:
                X_df (pd.DataFrame): Scaled X data.
                Y_df (pd.DataFrame): Scaled Y data.
                scaled_X_keys (list of str): Scaled X keys.
                scaled_Y_keys (list of str): Scaled Y keys.
        """        

        scaler = normalize_new.Scaler()
        X_df = scaler.scale_arrays(dataset, X_keys, X_maxmean_dic)
        Y_df = scaler.scale_arrays(dataset, Y_keys, Y_maxmean_dic)
        scaled_X_keys = X_df.keys()
        scaled_Y_keys = Y_df.keys()

        return X_df, Y_df, scaled_X_keys, scaled_Y_keys
    

    def unscale(self, preds_scaled, true_scaled, scaled_Y_keys, Y_maxmean_dic):
        """
        Unscale the Y data (predicted and true) back to the original scale and variables.

            Parameters:
                preds_scaled (np.array): Predicted Y values in maxmean scale.
                true_scaled (np.array): True Y values in maxmean scale.
                scaled_Y_keys (list of str): Scaled Y keys.
                Y_maxmean_dic (dict): Dictionary of max and mean for Y variables.

            Returns:
                preds_origscale_dic (dict): Predicted Y values in original scale.
                true_origiscale_dic: True Y values in original scale.
        """

        scaler = normalize_new.Scaler()
        preds_origscale_dic = scaler.invscale_arrays(preds_scaled, scaled_Y_keys, Y_maxmean_dic)
        true_origscale_dic = scaler.invscale_arrays(true_scaled, scaled_Y_keys, Y_maxmean_dic)
    
        return preds_origscale_dic, true_origscale_dic


    def prepData(self, datafile, X_maxmean_dic, Y_maxmean_dic, X_keys, Y_keys, jn, mask_value):
        """
        Prepares data for training by performing a mean-max scaling and phi-encoding, and then splitting dataset into (time-stepped) jets, other, and y data.

            Parameters:
                datafile (str): File name (and path) for training dataset.
                xmm_file (str): Path and file name for the X_maxmean file to be used in scaling.
                X_keys (list of str): Names of the (original) input variables.
                ymm_file (str): Path and file name for the Y_maxmean file to be used in scaling.
                Y_keys (list of str): Names of the (original) output variables.
                jn (int): Number of jets we're training with.
                mask_value (int): Value to mask non-existent jets with.

            Returns:
                totalX_jets (np.array): Scaled, time-stepped jets.
                totalX_other (np.array): Other scaled input data.
                Y_total (np.array): Scaled output data.
                scaled_X_keys (list of str): Names of the (scaled) input variables.
                scaled_Y_keys (list of str): Names of the (scaled) output variables.
        """

    
        print('Preparing data...')
        
        with h5py.File(datafile,'r') as dataset:   # Only want the dataset open as long as we need it
            
            # Create the timestep builder while we still have the dataset open
            timestep_builder = shape_timesteps_new.Shape_timesteps(dataset, jn, mask_value)
            
            # Scales data set to be between -1 and 1, with a mean of 0, and encodes phi in other variables (e.g. px, py)
            X_df, Y_df, scaled_X_keys, scaled_Y_keys = self.scale(dataset, X_keys, Y_keys, X_maxmean_dic, Y_maxmean_dic)

        # Split up jets and other for X, and Y just all stays together
        totalX_jets, totalX_other = timestep_builder.reshape_X(X_df)
        Y_total = np.array(Y_df)
        
        return totalX_jets, totalX_other, Y_total, scaled_X_keys, scaled_Y_keys


class Training:
    """
    A class for training a machine learning model.
    
        Methods:
            get_lr_metric: Gets the decaying learning rate, for monitoring purposes.
            build: Builds the desired model to be trained.
            save: Saves the model itself, the training history, and plots of the training loss.
            train: Builds, trains, and saves the model.

    """

    def __init__(self, train_file, xmm_file, ymm_file, Epochs, Patience, pretrain_file=None, finetune=False):
        """
        Initializes a trainer object.

            Parameters:
                train_file (str): Name (and path) of the training data file.
                xmm_file (str): Name (and path) of the X maxmean file.
                ymm_file (str): Name (and path) of the Y maxmean file.
                Epochs (int): Number of Epochs to train on.
                Patience (int): Patience (max number of epochs with no improvements) to train with.
                
            Optional:
                pretrain_file (str): Name (and path) of file containing the jet pre-trained model (Default: None).
                finetune (bool): An option if you would like to fine-tune the jet pretrained weights in TRecNet+ttbar+JetPretrain (Default: False).

            Attributes:
                train_file (str): Name (and path) of the training data file.
                xmm_file (str): Name (and path) of the X maxmean file.
                ymm_file (str): Name (and path) of the Y maxmean file.
                pretrain_file (str): Name (and path) of file containing the jet pre-trained model (Default: None).
                Epochs (int): Number of Epochs to train on.
                Patience (int): Patience (max number of epochs with no improvements) to train with.
                finetune (bool): An option if you would like to fine-tune the jet pretrained weights in TRecNet+ttbar+JetPretrain (Default: False).
                X_keys (list of str): Keys for the (original scale) X variables.
                Y_keys (list of str): Keys for the (original scale) Y variables.
                X_scaled_keys (list of str): Keys for the (maxmean scale) X variables.
                Y_scaled_keys (list of str): Keys for the (maxmean scale) Y variables.
        """
        self.train_file = train_file
        self.xmm_file = xmm_file
        self.ymm_file = ymm_file
        self.pretrain_file = pretrain_file
        self.Epochs = Epochs
        self.Patience = Patience
        self.finetune = finetune
        self.X_keys = None
        self.Y_keys = None
        self.X_scaled_keys = None
        self.Y_scaled_keys = None

        # Want to make sure we've got GPU
        if tf.config.list_physical_devices('GPU')==[]:
            print("WARNING: Networks will be trained on CPU. Move into container to use GPU.") 



    def build(self, model_name, jets_shape, other_shape, Y_scaled_keys, mask_value, pretrain_model=None):
        """
        Builds the desired model to be trained.

            Parameters:
                model_name (str): Name of the model ('TRecNet', 'TRecNet+ttbar', 'TRecNet+ttbar+JetPretrain', or 'JetPretrainer').
                jets_shape (arr of int): Shape of the jet input.
                other_shape (arr of int): Shape of the other (lep+met) input.
                had_keys (arr of str): Hadronic variable keys.
                lep_keys (arr of str): Leptonic variable keys.
                ttbar_keys (arr of str): ttbar variable keys.

            Options:
                pretrain_model (keras model): Loaded jet pretrain model, which is needed for the 'TRecNet+ttbar+JetPretrain' model (default: None).

            Returns:
                model (keras model): Built model, ready to be trained.
        """

        print('Building model...')

        # This will help for dimension of output layers
        had_shape = sum('th_' in key or 'wh_' in key for key in Y_scaled_keys)
        lep_shape = sum('tl_' in key or 'wl_' in key for key in Y_scaled_keys)
        ttbar_shape = sum('ttbar_' in key for key in Y_scaled_keys)

        # Input layers
        jet_input = Input(shape=(jets_shape[1], jets_shape[2]),name='jet_input')
        other_input = Input(shape=(other_shape[1]),name='other_input')

        # Mask the jets that are just there for padding (do I want this to be what's put into the flattened jets? or no, since it's being reshaped?) and use some TDDense layers
        Mask = Masking(mask_value, name='masking_jets')(jet_input)
        Maskshape = Reshape((jets_shape[1], jets_shape[2]), name='reshape_masked_jets')(Mask)
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
            output = Dense(jets_shape[1], activation='sigmoid', name='jet_match_output')(PreDense2)

            # Putting the model together
            model = keras.models.Model(inputs=[jet_input, other_input], outputs=output)
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-2, decay_steps=10000,end_learning_rate=5e-4,power=0.25)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['mae','mse'])
            
        # Lots more layers for the actual training
        else:

            # Use sigmoid to give jets a weight
            if model_name=='TRecNet+ttbar+JetPretrain':
                pretrain_model.trainable = False                                      # Freezing the jet pretrain model (i.e. want to use the previously trained weights)
                pretrain = pretrain_model([jet_input,other_input], training=False)    # Putting the inputs into the pretrain model
                Shape_Dot = Reshape((-1,1), name='reshape')(pretrain)
            else:
                PreDense3 = Dense(jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(PreDense2)
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
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-3, decay_steps=10000,end_learning_rate=5e-5,power=0.25)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            model.compile(loss='mae', optimizer= optimizer, metrics=['mse'])

        return model 


    def save(self, Model):
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
        file.write("Training Data File: %s \n" % self.train_file)
        file.write("X Maxmean File: %s \n" % self.xmm_file)
        file.write("Y Maxmean File: %s \n" % self.ymm_file)
        file.write("X Keys: "+', '.join(self.X_keys)+'\n')
        file.write("X Scaled Keys: "+', '.join(self.X_scaled_keys)+'\n')
        file.write("Y Keys: "+', '.join(self.Y_keys)+'\n')
        file.write("Y Scaled Keys: "+', '.join(self.Y_scaled_keys)+'\n')
        file.write("Max Number of Epochs: %s \n" % self.Epochs)
        file.write("Number of Epochs Used: %s \n" % len(Model.training_history['loss']))
        file.write("Patience: %s \n" % self.Patience)
        file.write("Model Architecture:\n")
        Model.model.summary(expand_nested=True, show_trainable=True, print_fn=lambda x: file.write(x + '\n'))
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


        # Create an object to use utilities
        processor = Utilities()

        # These are the keys for what we're feeding into the pre-processing, and getting back in the end
        # X and Y variables to be used (NOTE: later have option to feed these in?)
        self.X_keys, self.Y_keys = processor.getInputKeys(Model.model_name, Model.n_jets)

        # Load maxmean
        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)

        # Pre-process the data
        totalX_jets, totalX_other, totalY, self.X_scaled_keys, self.Y_scaled_keys = processor.prepData(self.train_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, Model.n_jets, Model.mask_value)

        # Set how the data will be split (70 for training, 15 for validation, 15 for testing)
        split = 70/85   # Gave 85% to train file, now want 70% for the actual training
        trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = train_test_split(totalX_jets, totalX_other, totalY, train_size=split)



        # Need to load jet pretraining if TRecNet+ttbar+JetPretrain
        if Model.model_name=='TRecNet+ttbar+JetPretrain':
            if self.pretrain_file==None:
                print('Please provide a jet pretrained model in order to train TRecNet+ttbar+JetPretrain.')
                sys.exit()
            pretrain_model = keras.models.load_model(self.pretrain_file)
            Model.model = self.build(Model.model_name, trainX_jets.shape, trainX_other.shape, self.Y_scaled_keys, Model.mask_value, pretrain_model = pretrain_model)
        else:
            Model.model = self.build(Model.model_name, trainX_jets.shape, trainX_other.shape, self.Y_scaled_keys, Model.mask_value)
        print(Model.model_name+' model has been built and compiled.')

        # Set early stopping (so no overfitting) and tensorboard callback (for monitoring)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.Patience)  # patience=4 means stop training after 4 epochs with no improvement
        log_dir = "tensorboard_logs/fit/"+Model.model_id
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



        # Fit/train the model      
        history = Model.model.fit([trainX_jets, trainX_other], trainY, verbose=1, epochs=self.Epochs, validation_data=([valX_jets, valX_other], valY), shuffle=True, callbacks=[early_stop, tensorboard_callback], batch_size=1000)
        Model.training_history = history.history

        # Save the model, its history, and loss plots
        self.save(Model)
    
    
        
        # Need to unfreeze jet pretraining and recompile TRecNet+ttbar+JetPretrain if we want to unfreeze the weights
        if Model.model_name=='TRecNet+ttbar+JetPretrain':
            
            print('Finetuning...')
            
            # Copy the old model (just alter the name and id)
            Unfrozen_Model = TRecNet_Model(model_name=Model.model_name+'Unfrozen')
            Unfrozen_Model.model = Model.model
            Unfrozen_Model.frozen_model_id = Model.model_id
            Unfrozen_Model.n_jets = Model.n_jets
            del Model
            
            # Double check that we're unfreezing the correct layer
            if Unfrozen_Model.model.layers[4].trainable:
                print('Trying to fine-tune the wrong layer. Look at model summary below and find the correct index for the jet pretrained layer.')
                print(Unfrozen_Model.model.summary(expand_nested=True, show_trainable=True))
                sys.exit()

            # Unfreeze ALL jet pretrained layers
            Unfrozen_Model.model.layers[4].trainable = True    
            
            # Use a smaller learning rate since we're fine-tuning now
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-3, decay_steps=10000,end_learning_rate=5e-5,power=0.25)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            
            # Recompile the model    
            Unfrozen_Model.model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])
                
            # Set early stopping (so no overfitting) and tensorboard callback (for monitoring)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.Patience)  # patience=4 means stop training after 4 epochs with no improvement
            log_dir = "tensorboard_logs/fit/"+Unfrozen_Model.model_id
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


                
            # Re-fit/train the model      
            history = Unfrozen_Model.model.fit([trainX_jets, trainX_other], trainY, verbose=1, epochs=self.Epochs, validation_data=([valX_jets, valX_other], valY), shuffle=True, callbacks=[early_stop, tensorboard_callback], batch_size=1000)
            Unfrozen_Model.training_history = history.history
            
            # Save the model, its history, and loss plots
            self.save(Unfrozen_Model)





class Validating:
    
    def __init__(self, train_file, xmm_file, ymm_file):
        self.train_file = train_file
        self.xmm_file = xmm_file
        self.ymm_file = ymm_file
        
    
    def plot(self, var, preds, truths, save_loc):
        
        nbins = 30
        max_val = max(max(preds), max(truths))
        min_val = min(min(preds), min(truths))
        
        plt.figure()
        plt.hist(truths, bins=nbins, range=(min_val, max_val), histtype='step',label='truth')
        plt.hist(preds, bins=nbins, range=(min_val, max_val), histtype='step',label='reco')
        plt.legend()
        plt.xlabel(var)
        plt.ylabel('Counts')
        plt.savefig(save_loc+var)
        
        
        
    def validate(self, Model):
    
        # Create an object to use utilities
        processor = Utilities()
        
        # Load the model
        Model.model = keras.models.load_model(Model.model_name+'/'+Model.model_id+'/'+Model.model_id+'.keras')

        # These are the keys for what we're feeding into the pre-processing, and getting back in the end
        # X and Y variables to be used (NOTE: later have option to feed these in?)
        self.X_keys, self.Y_keys = processor.getInputKeys(Model.model_name, Model.n_jets)

        # Load maxmean
        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)

        # Pre-process the data
        totalX_jets, totalX_other, totalY, self.X_scaled_keys, self.Y_scaled_keys = processor.prepData(self.train_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, Model.n_jets, Model.mask_value)

        # Set how the data will be split (70 for training, 15 for validation, 15 for testing)
        split = 70/85   # Gave 85% to train file, now want 70% for the actual training
        trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = train_test_split(totalX_jets, totalX_other, totalY, train_size=split)
        
        # Predictions and truth BEFORE they're back to the original scale
        print('Making predictions ...')
        preds_scaled = Model.model.predict([valX_jets, valX_other])
        true_scaled = valY 
        
        # Invert scaling
        preds_origscale_dic, true_origscale_dic = processor.unscale(preds_scaled, true_scaled, self.Y_scaled_keys, Y_maxmean_dic)
        
        # Create basic plots of results and save them
        # Create directory for saving things in if it doesn't exist
        save_loc = Model.model_name+'/'+Model.model_id+'/val_plots/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)
        for key in preds_origscale_dic.keys():
            self.plot(key, preds_origscale_dic[key], true_origscale_dic[key], save_loc)
    







class Testing:

    def __init__(self, test_file, xmm_file, ymm_file, data_type):
        self.test_file = test_file
        self.xmm_file = xmm_file
        self.ymm_file = ymm_file
        self.data_type = data_type
        self.X_keys = None
        self.Y_keys = None
        self.X_scaled_keys = None
        self.Y_scaled_keys = None

    def calculateExtraVariables(self, model_name, preds, eventNumbers):

        print('Calculating extra variables ...')
        
        # Create the 4-vectors for hadronic and leptonic tops
        th_vec = vector.arr({"pt": preds['th_pt'], "phi": preds['th_phi'], "eta": preds['th_eta'],"mass": preds['th_m']})
        tl_vec = vector.arr({"pt": preds['tl_pt'], "phi": preds['tl_phi'], "eta": preds['tl_eta'],"mass": preds['tl_m']}) 
        
        # Create 4-vector for ttbar (and get ttbar variables if necessary)
        if '+ttbar' in model_name:
            ttbar_vec = vector.arr({"pt": preds['ttbar_pt'], "phi": preds['ttbar_phi'], "eta": preds['ttbar_eta'],"mass": preds['ttbar_m']})
        else:
            ttbar_vec = th_vec + tl_vec
            preds['ttbar_pt'] = ttbar_vec.pt
            preds['ttbar_phi'] = ttbar_vec.phi
            preds['ttbar_eta'] = ttbar_vec.eta
            preds['ttbar_m'] = ttbar_vec.m
        
        # Get vector package to calculate the energies
        preds['th_E'] = th_vec.energy
        preds['tl_E'] = tl_vec.energy
        preds['ttbar_E'] = ttbar_vec.energy
        
        # Get vector package to calculate the rapidities
        preds['th_y'] = th_vec.rapidity
        preds['tl_y'] = tl_vec.rapidity
        preds['ttbar_y'] = ttbar_vec.rapidity
        
        # Calculate pout for th and tl
        ez = vector.arr({"px":np.array([0]*preds['th_eta'].shape[0]),"py":np.array([0]*preds['th_eta'].shape[0]),"pz":np.array([1]*preds['th_eta'].shape[0])})
        thP_vec = vector.arr({"px":th_vec.px,"py":th_vec.py,"pz":th_vec.pz})
        tlP_vec = vector.arr({"px":tl_vec.px,"py":tl_vec.py,"pz":tl_vec.pz})
        preds['th_pout'] = -thP_vec.dot(tlP_vec.cross(ez)/(tlP_vec.cross(ez).mag))   # For some reason this calculation is off by a negative compared to the truth ...
        preds['tl_pout'] = -tlP_vec.dot(thP_vec.cross(ez)/(thP_vec.cross(ez).mag))

        # Calculate some other ttbar variables
        preds['ttbar_dphi'] = abs(th_vec.deltaphi(tl_vec))
        preds['ttbar_Ht'] = preds['th_pt']+preds['tl_pt']
        preds['ttbar_yboost'] = 0.5*(preds['th_y']+preds['tl_y'])
        preds['ttbar_ystar'] = 0.5*(preds['th_y']-preds['tl_y'])
        preds['ttbar_chi'] = np.exp(2*np.abs(preds['ttbar_ystar']))
        
        preds['eventNumber'] = eventNumbers

        return preds
    


    def save(self, Model, preds_df, truths_df, save_loc):

        print('Saving results ...')

        save_name = save_loc+Model.model_id+'_'+self.data_type+'_Results.root'
        results_file = uproot.recreate(save_name)

        if self.data_type=='nominal': 
            results_file["reco"] = {key:preds_df[key] for key in list(preds_df.keys())}
            results_file["parton"] = {key:truths_df[key] for key in list(truths_df.keys())}
        else:
            results_file[self.data_type] = {key:preds_df[key] for key in list(preds_df.keys())}

        print('Results saved in %s.' % save_name)




    def test(self, Model, save_loc):

        # Create an object to use utilities
        processor = Utilities()
        
        

        # Load the things we'll need
        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)
        Model.model = keras.models.load_model(Model.model_name+'/'+Model.model_id+'/'+Model.model_id+'.keras')


        # These are the keys for what we're feeding into the pre-processing, and getting back in the end
        # X and Y variables to be used (NOTE: later have option to feed these in)
        self.X_keys, self.Y_keys = processor.getInputKeys(Model.model_name,Model.n_jets)

        # Pre-process the data
        testX_jets, testX_other, testY, self.X_scaled_keys, self.Y_scaled_keys = processor.prepData(self.test_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, Model.n_jets, Model.mask_value)


        # Predictions and truth BEFORE they're back to the original scale
        preds_scaled = Model.model.predict([testX_jets, testX_other])
        true_scaled = testY 

        # Invert scaling
        preds_origscale_dic, true_origscale_dic = processor.unscale(preds_scaled, true_scaled, self.Y_scaled_keys, Y_maxmean_dic)



        # Get the true values
        with h5py.File(self.test_file,'r') as dataset:
            eventNumbers = np.array(dataset.get('eventNumber'))
            if self.data_type=='nominal': 
                truth_keys = ['eventNumber','jet_n']+list(filter(lambda a: 'th_' in a or 'tl_' in a or 'ttbar_' in a or 'wh_' in a or 'wl_' in a, dataset.keys()))  
                truths_df = pd.DataFrame({key:dataset.get(key) for key in truth_keys})
            else:
                truths_df = None  # If working with systematics, we have no truth
                
        # Calculate all the same variables we had for truth, using the predicted values
        preds_df = pd.DataFrame(preds_origscale_dic)
        preds_df = self.calculateExtraVariables(Model.model_name, preds_df, eventNumbers)

        # Save results
        self.save(Model, preds_df, truths_df)

        
        





# ------ MAIN CODE ------ #

# Set up argument parser
parser = ArgumentParser()

# These arguments we want to be able to use for all of them
parent_parser = ArgumentParser(add_help=False)
parent_parser.add_argument('--model_name', help='Name of the model to be trained.', type=str, required=True, choices=['TRecNet','TRecNet+ttbar','TRecNet+ttbar+JetPretrain','JetPretrainer'])
parent_parser.add_argument('--data', help="Path and file name for the training data to be used (e.g. '/mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_6j_train.h5').", type=str, required=True)
parent_parser.add_argument('--xmaxmean', help="Path and file name for the X_maxmean file to be used (e.g. '/home/jchishol/TRecNet/X_maxmean_variables_ttbar_ljets_6j_train.npy')", type=str, required=True)
parent_parser.add_argument('--ymaxmean', help="Path and file name for the Y_maxmean file to be used (e.g. '/home/jchishol/TRecNet/Y_maxmean_variables_ttbar_ljets_6j_train.npy')", type=str, required=True)

# Get the model argument so it can be used for conditionals
model_arg = parent_parser.parse_known_args()[0].model_name


# Set up subparsers
subparsers = parser.add_subparsers(dest='module', help='Module you would like to use.', required=True)
training_parser = subparsers.add_parser('training', parents=[parent_parser])
validating_parser = subparsers.add_parser('validating', parents=[parent_parser])
testing_parser = subparsers.add_parser('testing', parents=[parent_parser])

# These arguments we need specifically for training
training_parser.add_argument('--epochs', help='Maximum number of epochs to train on (default: 256).', type=int, default=256)
training_parser.add_argument('--patience', help='Number of epochs without improvement before training is stopped early (default: 4).', type=int, default=4)
if model_arg=='TRecNet+ttbar+JetPretrain':
    training_parser.add_argument('--jet_pretrain_model', help="Path and file name of jet pre-trained keras model.", required=True)
else:
    training_parser.add_argument('--jn', help="Number of jets to input to the neural network.", type=int, default=6)


# These arguments we need specifically for validating
validating_parser.add_argument('--model_id', help="Unique save name for the trained model.", type=str, required=True)

# These arguments we need specifically for testing
testing_parser.add_argument('--model_id', help="Unique save name for the trained model.", type=str, required=True)
testing_parser.add_argument('--data_type', choices=['nominal','sysUP','sysDOWN'],help='Type of data.', required=True)
testing_parser.add_argument('--save_loc', help="Directory in which to save the results (e.g. '/mnt/xrootdg/jchishol/mntuples_08_01_22/Results/').", type=str, required=True)

# Parse arguments
args = parser.parse_args()


# Do the machine learning stuff now
print('Beginning '+args.module+' for '+args.model_name+'...')

if args.module=='training':
    
    if args.model_name == 'TRecNet+ttbar+JetPretrain':
        Model = TRecNet_Model(args.model_name, n_jets=int(args.jet_pretrain_model.split('/')[-1].split('jets')[0].split('_')[-1]))
        Trainer = Training(args.data, args.xmaxmean, args.ymaxmean, args.epochs, args.patience, pretrain_file=args.jet_pretrain_model)
        Trainer.train(Model)
    else:
        Model = TRecNet_Model(args.model_name, n_jets=args.jn)
        Trainer = Training(args.data, args.xmaxmean, args.ymaxmean, args.epochs, args.patience)
        Trainer.train(Model)
        
elif args.module=='validating':
    jn = args.model_name.split('jets')[0].split('_')[-1]
    Model = TRecNet_Model(args.model_name, model_id=args.model_id,n_jets=jn)
    Validator = Validating(args.data, args.xmaxmean, args.ymaxmean)
    Validator.validate(Model)
    
else:
    jn = args.model_name.split('jets')[0].split('_')[-1]
    Model = TRecNet_Model(args.model_name, model_id=args.model_id,n_jets=jn)
    Tester = Testing(args.data, args.xmaxmean, args.ymaxmean, args.data_type)
    Tester.test(Model, args.save_loc)




print('done :)')


