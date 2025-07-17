##########################################################################
#                                                                        #
#  ModelBuilder.py                                                       #
#  Author: Jenna Chisholm                                                #
#  Updated: Jul.3/25                                                     #
#                                                                        #
#  Defines classes and functions relevant for to building TRecNet        #
#  models.                                                               # 
#                                                                        #
#  Thoughts for improvements:                                            #
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


class TRecNet_Model:
    """
    A class for creating a machine learning model object, mainly to store relevant attributes of the model.
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
                mask_value (int): 
                n_jets (int): Number of jets the model is trained with.
        """
        
        self.model_name = model_name
        self.model_id = time.strftime(model_name+"_"+str(n_jets)+"jets_%Y%m%d_%H%M%S") if model_id==None else model_id   # Model unique save name (based on the date)
        self.n_jets = n_jets if model_id==None else int(model_id.split('_')[1].split('jets')[0]) # If not given, get from model_id
        self.mask_value = -2   # Define here so it's consist between model building and jet timestep building
        
        self.jets_shape = None
        self.other_shape = None
        self.had_shape = None
        self.lep_shape = None
        self.ttbar_shape = None
        self.bbbar_shape = None
        
        

class ModelBuilder:
    """
    A class for building TRecNet models.
    
        Parameters:
            Model (TRecNet_Model): TRecNet_Model object that stores important parameters of the model's architecture.
    """  
    
    def __init__(self, Model):
        self.Model = Model
        
    def construct_input_layers(self):
        
        jet_input = Input(shape=(self.Model.jets_shape[1], self.Model.jets_shape[2]),name='jet_input')
        other_input = Input(shape=(self.Model.other_shape[1],),name='other_input')
        
        return jet_input, other_input
    
    
    def get_initial_jet_processor_output(self, jet_input):
        
        Mask = Masking(self.Model.mask_value, name='masking_jets')(jet_input)
        Maskshape = Reshape((self.Model.jets_shape[1], self.Model.jets_shape[2]), name='reshape_masked_jets')(Mask)
        TDDense11 = TimeDistributed(Dense(128, activation='relu'), name='TDDense128')(Maskshape)
        TDDense12 = TimeDistributed(Dense(64, activation='relu'), name='TDDense64')(TDDense11)
        
        return TDDense12
        
    # Concatenate flattened jets and other and use some dense layers (but not for TRecNet+ttbar+JetPretrain, since this is done in JetPretrainer)
    def get_jet_classifier_output(self, jet_input, other_input):
        
        flat_jets =  Flatten(name ='flattened_jets')(jet_input) 
        concat0 = concatenate([other_input, flat_jets], name = 'concat_jets_other')
        PreDense1 = Dense(256, activation='relu', name = 'dense256_1')(concat0)
        PreDense2 = Dense(256, activation='relu', name = 'dense256_2')(PreDense1) 
        
        Sigmoid_output = Dense(self.Model.jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(PreDense2)
            
        return Sigmoid_output
    
    def get_initial_lepmet_processor_output(self, other_input):
    
        Dense21 = Dense(128, activation='relu', name='dense128')(other_input)
        Dense22 = Dense(64, activation='relu', name='dense64')(Dense21)
        flat_other = Flatten(name='flattened_other')(Dense22)
        
        return flat_other
    
    def get_weighted_jets(self, in_weights_layer, in_jets_layer):
        
        # Use some more TDDense layers with the weighted jets
        Shape_Dot = Reshape((-1,1), name='reshape')(in_weights_layer)
        weighted_jets = Multiply(name='weight_jets')([Shape_Dot, in_jets_layer])
        
        return weighted_jets
        
    
    def get_weighted_jet_processor_output(self, weighted_jets_layer):
        
        TDDense13 = TimeDistributed(Dense(256, activation='relu'), name='TDDense256_1')(weighted_jets_layer)
        TDDense14= TimeDistributed(Dense(256, activation='relu'), name='TDDense256_2')(TDDense13)
        Flat_wjets = Flatten(name='flattened_weighted_jets')(TDDense14)
        
        return Flat_wjets
    
    def get_lep_output(self, concat_layer):
        
        ldense1 = Dense(256, activation='relu', name='ldense256')(concat_layer)
        ldense2 = Dense(128, activation='relu', name='ldense128')(ldense1)
        loutput = Dense(self.Model.lep_shape, name='lep_output')(ldense2)
        
        return loutput
    
    def get_had_output(self, loutput, concat_layer):
        
        hconcat = concatenate([loutput, concat_layer])
        hdense1 = Dense(256, activation='relu', name='hdense256')(hconcat)
        hdense2 = Dense(128, activation='relu', name='hdense128')(hdense1)
        houtput = Dense(self.Model.had_shape+self.Model.ttbar_shape, name='had_output')(hdense2)
        
        return houtput
    
    def get_bbbar_output(self, loutput, houtput, wjets):
        
        jbdense1 = TimeDistributed(Dense(256, activation='relu'), name='jb_TDDense256_1')(wjets)
        jbdense2 = TimeDistributed(Dense(256, activation='relu'), name='jb_TDDense256_2')(jbdense1)
        jbflatten = Flatten(name='jb_flattened')(jbdense2)
        
        bconcat = concatenate([loutput, houtput, jbflatten])
        bdense1 = Dense(256, activation='relu', name='bdense256_1')(bconcat)
        bdense2 = Dense(256, activation='relu', name='bdense256_2')(bdense1)
        bdense3 = Dense(128, activation='relu', name='bdense128_3')(bdense2)
        boutput = Dense(self.Model.bbbar_shape, name='b_bbar_output')(bdense3)
        
        return boutput
    
    def get_all_output(self, concat, wjets):
        
        if ('ttbb' in self.Model.name):
            
            loutput = self.get_lep_output(concat)
            houtput = self.get_had_output(loutput, concat)
            boutput = self.get_bbbar_output(loutput, houtput, wjets)
            
            output = concatenate([houtput, loutput, boutput], name='output')
            
        else:
            
            loutput = self.get_lep_output(concat)
            houtput = self.get_had_output(loutput, concat)
            
            output = concatenate([houtput, loutput], name='output')
            
        return output
        
        
        
    def construct_architecture(self, jet_input, other_input, pretrain_model=None):
        """
        Parameters:
                jet_input (keras.layers.Input): Input layer for jet observables.
                other_input (keras.layers.Input): Output layer for other (lep, met) observables.
            
            Optional:
                pretrain_model (Model object): Jet pre-trained model (default: None).

            Returns:
                model (Model object): Built model.

        """

        print("Building model "+self.Model.model_name+"...\n")
        
        # Jet Pretrainer build
        if self.Model.model_name=='JetPretrainer':
            
            output = self.get_jet_classifier_output(jet_input, other_input)
            
        
        # TRecNet and TRecNet+ttbar build  (they're effectively the same)
        elif self.Model.model_name=='TRecNet' or self.Model.model_name=='TRecNet+ttbar':
            
            # Create initial jet processor and lep+met processors
            TDDense12 = self.get_initial_jet_processor_output(jet_input)
            flat_other = self.get_initial_lepmet_processor_output(other_input)
            
            # Create jet classifier and get sigmoid outputs
            PreDense3 = self.get_jet_classifier_output(jet_input, other_input)
            
            # Create weighted jet processor
            wjets = self.get_weighted_jets(PreDense3, TDDense12)
            Flat_wjets = self.get_weighted_jet_processor_output(wjets)
            
            # Concatenate the two sides
            concat = self.concatenate([flat_other, Flat_wjets], name = 'concat_everything')
            
            # Get ouput layer
            output = self.get_all_output(concat, wjets)
        
        # TRecNet+ttbb+JetPretain build
        elif self.Model.model_name=='TRecNet+ttbb+JetPretrain':
            
            # Create initial jet processor and lep+met processors
            TDDense12 = self.get_initial_jet_processor_output(jet_input)
            flat_other = self.get_initial_lepmet_processor_output(other_input)
            
            # Using pretrained jet classifier instead of training it here
            pretrain_model.trainable = False                                      # Freezing the jet pretrain model (i.e. want to use the previously trained weights)
            pretrain = pretrain_model([jet_input,other_input], training=False)    # Putting the inputs into the pretrain model

            # Create weighted jet processor
            wjets = self.get_weighted_jets(pretrain, TDDense12)
            Flat_wjets = self.get_weighted_jet_processor_output(wjets)
            
            # Concatenate the two sides
            concat = concatenate([flat_other, Flat_wjets], name = 'concat_everything')
            
            # Get ouput layer
            output = self.get_all_output(concat, wjets)
            
            
        return output # need to return ouptut (do rest of model somewhere else?)
    
    
    
    def create_model(self, initial_lr, final_lr_div, lr_power, lr_decay_step, pretrain_model=None, frozen_file=None):
        
        # For TRecNet+ttbar+JetPretrainUnfrozen we read in the model, but for others we construct it
        if self.Model.model_name == 'TRecNet+ttbar+JetPretrainUnfrozen':
            
            # Load the frozen model to start with
            model = keras.layers.TFSMLayer(frozen_file, call_endpoint="serving_default")
            
            # Find the jet pre-training layer and unfreeze all those sublayers
            for layer in model.layers:
                if isinstance(layer, keras.Model):
                    layer.trainable = True 
            
        else:
       
            # Construct the model's architecture
            jet_input, other_input = self.construct_input_layers()
            output = self.construct_architecture(jet_input, other_input, pretrain_model)
            model = keras.models.Model(inputs=[jet_input, other_input], outputs=output)
             
             
        # Learning rate and optimization settings
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr, decay_steps=lr_decay_step,end_learning_rate=initial_lr/final_lr_div,power=lr_power)
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile with relevant loss functions
        if self.Model.model_name == 'JetPretrainer':
            model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['mae','mse'],jit_compile=False)
        else:
            model.compile(loss='mae', optimizer= optimizer, metrics=['mse'],jit_compile=False)
                

        return model 
