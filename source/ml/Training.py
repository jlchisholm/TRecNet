##########################################################################
#                                                                        #
#  Training.py                                                           #
#  Author: Jenna Chisholm                                                #
#  Updated: Sept.8/25                                                    #
#                                                                        #
#  Defines classes and functions relevant for training and hypertuning   #
#  neural networks.                                                      # 
#                                                                        #
#  Thoughts for improvements: Have X and Y keys as input variables?      #
#                                                                        #
##########################################################################


import os, sys, shutil
from datetime import datetime
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.environ["CUDA_VISIBLE_DEVICES"]="2"    # These are the GPUs visible for training
from argparse import ArgumentParser
from contextlib import redirect_stdout
import uproot
import h5py
import json
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
import keras_tuner as kt
# get path utility
from paths import tb_fit_dir
#from clr_callback import * 

from MLUtil import Utilities
from Scaler import Scaler
from TRecNet_Model import TRecNet_Model
from ModelBuilder import ModelBuilder

import tracemalloc
tracemalloc.start()

# needed to add this to get it to work but need to make it somethinf the user can change so not hard coded
IMPORTANT_TARGETS = [
    # hadronic top
    "th_pt", "th_eta", "th_phi", "th_m",
    # hadronic W
    "wh_pt", "wh_eta", "wh_phi", "wh_m",
    # leptonic top
    "tl_pt", "tl_eta", "tl_phi", "tl_m",
    # leptonic W
    "wl_pt", "wl_eta", "wl_phi", "wl_m",
    # ttbar system
    "ttbar_pt", "ttbar_eta", "ttbar_phi", "ttbar_m",
    # bb system (your v5x1 b1/b2)
    "b1_pt", "b1_eta", "b1_phi", "b1_m",
    "b2_pt", "b2_eta", "b2_phi", "b2_m",
]

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
        self.split = config_file["split"][0]/(config_file["split"][0]+config_file["split"][1])   # [0]=% in train, [1]=% in val, [2]=% in test --> to get percentage of train file to actually use for training, need train / train + val
        
        self.max_epochs = config_file["max_epochs"]
        self.patience = config_file["patience"]
        # need indexes for shuffling in cv
        self.train_idx = None
        self.val_idx = None
        
        self.train_size = 0
        self.val_size = 0
        
        self.X_keys = None
        self.Y_keys = None
        self.X_scaled_keys = None
        self.Y_scaled_keys = None
        
        self.training_time = None
        self.training_history = None
        # need to be able to see this in the model for cv and learning curve
        self.hypertune_data_frac = 1.0
        self.config = config_file
        # I thought this was redundant but doesnt work if I dont put it here
        # i think the it needs to be instantiated or something which doesnt make sense for python 
        # but alas the computer knows better than me 
        self.arch_hparams = config_file.get("hparams", {}) or {}

        # Want to make sure we've got GPU
        if tf.config.list_physical_devices('GPU')==[]:
            print("WARNING: Networks will be trained on CPU. Move into container to use GPU.") 
          
          
    def set_create_params(self, create_config):
        """
        Sets the training parameters from the config file.

            Parameters:
                create_config (dict): Creation configuration dictionary.
        """
        
        self.jet_pretrain_file = create_config["pretrained_jet_classifier"]
        self.bb_pretrain_file = create_config["pretrained_bb_classifier"]
        
    
    def set_unfreeze_params(self, unfreeze_config, Model):
        """
        Sets the training parameters from the config file.

            Parameters:
                unfreeze_config (dict): Unfreeze configuration dictionary.
        """
        
        Model.unfreeze = True
        self.frozen_model = unfreeze_config["frozen_model"]
        self.frozen_model_id = os.path.basename(self.frozen_model)
        self.frozen_file = unfreeze_config.get("frozen_file", self.frozen_model)
         
            
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
        self.optimizer = train_config.get("optimizer", "adam") # added optimizer option, default to adam if not specified
        
        
    def set_hyper_config(self, hyper_config):
        """
        Sets the hypertuning configuration for later use.

            Parameters:
                hyper_config (dict): Hypertuning configuration dictionary.
        """
        
        self.tuner_type = hyper_config["tuner"]
        self.hyper_config = hyper_config["hyperparams"]
        # need this for arch hyperparams
        self.arch_hyper_config = hyper_config.get("arch_hyperparams", {}) or {}
        self.hypertune_data_frac = float(hyper_config.get("data_frac", 1.0))
 
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
        self.X_keys, self.Y_keys = processor.getInputKeys(Model.model_v, Model.n_jets, Model.with_ttbar, Model.b_mode)

        # Load maxmean
        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)

        # Pre-process the data
        totalX_jets, totalX_other, totalY, self.X_scaled_keys, self.Y_scaled_keys = processor.scale_and_shape(self.train_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, Model.n_jets, Model.mask_value)
        
        # Save shapes for later
        Model.had_shape = sum('th_' in key or 'wh_' in key for key in self.Y_scaled_keys)
        Model.lep_shape = sum('tl_' in key or 'wl_' in key for key in self.Y_scaled_keys)
        Model.ttbar_shape = sum('ttbar_' in key for key in self.Y_scaled_keys)
        Model.bbbar_shape = sum('b_' in key or 'bbar_' in key or 'b1_' in key or 'b2_' in key for key in self.Y_scaled_keys)

        
        # I was trying to implement a sampler but gave up but this is still useful
        # so instead of splitting the data, we split an array of indexes, and then use those to index into the data arrays

        # total number of events
        N = len(totalY)
        all_idx = np.arange(N, dtype=np.int64)

        # split indexes (avoid leakage; use fixed seed for reproducibility)
        train_idx, val_idx = train_test_split(
            all_idx,
            train_size=self.split,
            shuffle=True,
            random_state=42,
        )
        frac = float(getattr(self, "hypertune_data_frac", 1.0))
        if frac < 1.0:
            orig_n = len(train_idx)
            rng = np.random.RandomState(123)  # fixed for reproducibility
            m = max(1, int(round(frac * len(train_idx))))
            m = min(m, len(train_idx))
            train_idx = rng.choice(train_idx, size=m, replace=False)
            print(f"[load_and_prep] Hypertune data_frac={frac:.3f} -> using {m}/{orig_n} train events")

        self.train_idx = train_idx
        self.val_idx = val_idx

        # Create the training and validation sets
        trainX_jets  = totalX_jets[train_idx]
        valX_jets    = totalX_jets[val_idx]
        trainX_other = totalX_other[train_idx]
        valX_other   = totalX_other[val_idx]
        trainY       = totalY[train_idx]
        valY         = totalY[val_idx]
        
        # Save the number of events being used (to be written out later)
        self.train_size = len(trainY)
        self.val_size = len(valY)
        
        # Save the shapes for later
        Model.jets_shape = trainX_jets.shape
        Model.other_shape = trainX_other.shape
        
        return trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY
 

    def _resolve_run_hparams(self, Model):
        '''internal method to make a dict of hparams for the run in hparam optimization'''
        arch = dict(getattr(Model, "hparams", {}) or {})

        run_hp = {
            # identity
            "model_v": getattr(Model, "model_v", None),
            "model_name": getattr(Model, "model_name", None),
            "model_id": getattr(Model, "model_id", None),

            # training hyperparams
            "initial_lr": float(self.initial_lr),
            "final_lr_div": int(self.final_lr_div),
            "lr_power": float(self.lr_power),
            "lr_decay_step": int(self.lr_decay_step),
            "batch_size": int(self.batch_size),
            "optimizer": str(getattr(self, "optimizer", "adam")),

            # schedule info 
            "end_lr": float(self.initial_lr / float(self.final_lr_div)),

            # architecture hyperparams
            "arch": arch,
        }
        return run_hp
    

    def save_model(self, Model):
        """
        Saves the model itself, the training history, and plots of the training loss.

            Parameters:
                Model (Model object): Model to be saved.

            Returns:
                Layout:
                trained_models/<Model.model_v>/<Model.model_name>/<Model.model_id>/
                    model/   <model_id>.keras
                    history/ <model_id>_TrainHistory.npy
                    scaling/ X_maxmean_*.npy, Y_maxmean_*.npy
                    info/    run_Info.txt
                    plots/
                    train/
                        <model_id>_MAE.png
                        <model_id>_MSE.png
                        (and BCE plot if pretrainers are used)
        """

        #  check if the model is in cv mode so it saves folds under same folder

        if hasattr(Model, "run_dir_override"):
            run_dir = Model.run_dir_override
        else:
            # canonical run directory 
            run_dir = os.path.join('trained_models', Model.model_v, Model.model_name, Model.model_id)

        #  create subfolders 
        sub_model   = os.path.join(run_dir, 'model')
        sub_hist    = os.path.join(run_dir, 'history')
        sub_scale   = os.path.join(run_dir, 'scaling')
        sub_info    = os.path.join(run_dir, 'info')
        sub_plots   = os.path.join(run_dir, 'plots', 'train')

        # make the dirs
        for d in (sub_model, sub_hist, sub_scale, sub_info, sub_plots):
            os.makedirs(d, exist_ok=True)

        # Save model and history
        model_path = os.path.join(sub_model, f'{Model.model_id}.keras') # get keras path
        hist_path  = os.path.join(sub_hist,  f'{Model.model_id}_TrainHistory.npy') # get history path
        Model.model.save(model_path) # actual model save
        np.save(hist_path, self.training_history) # save hist
        #  Save maxmean scaling files
        if os.path.isfile(self.xmm_file): shutil.copy(self.xmm_file, sub_scale)
        if os.path.isfile(self.ymm_file): shutil.copy(self.ymm_file, sub_scale)

        # Save important information about this model into a text file
        info_path = os.path.join(sub_info, 'run_Info.txt')
        with open(info_path, "w") as file:
            file.write(f"Model ID: {Model.model_id}\n")
            if Model.unfreeze: file.write("Frozen Model ID: %s \n" % self.frozen_model_id)
            if Model.use_JetPretraining: file.write("JetPretrain Model File: %s \n" % self.jet_pretrain_file)
            if Model.use_bbPretraining: file.write("bbPretrain Model File: %s \n" % self.bb_pretrain_file)
            file.write("---------------------------------------------------\n")
            file.write("Training Data File: %s \n" % self.train_file)
            file.write("Percentage of Train Data Used for Training: %s \n" % self.split)     
            file.write("Percentage of Train Data Used for Validation: %s \n" % (1 - self.split))   
            file.write("X Maxmean File: %s \n" % self.xmm_file)
            file.write("Y Maxmean File: %s \n" % self.ymm_file)
            file.write("---------------------------------------------------\n")
            file.write("Number of training events: %s \n" % self.train_size)
            file.write("Number of validating events: %s \n" % self.val_size)
            file.write("---------------------------------------------------\n")
            file.write("X Keys: "+', '.join(self.X_keys)+'\n')
            file.write("X Scaled Keys: "+', '.join(self.X_scaled_keys)+'\n')
            file.write("Y Keys: "+', '.join(self.Y_keys)+'\n')
            file.write("Y Scaled Keys: "+', '.join(self.Y_scaled_keys)+'\n')
            file.write("---------------------------------------------------\n")
            file.write("Learning Rate: Polynomial Decay with initial_lr=%s, final_lr=%s, decay_step=%s, and power=%s \n" % (self.initial_lr, (self.initial_lr/self.final_lr_div), self.lr_decay_step, self.lr_power))
            file.write("Batch Size: %s \n" % self.batch_size)
            file.write("Max Number of Epochs: %s \n" % self.max_epochs)
            file.write("Number of Epochs Used: %s \n" % len(self.training_history['loss']))
            file.write("Patience: %s \n" % self.patience)
            file.write("Training Time: %02d:%02d:%02d:%02d \n" % (self.training_time.days, self.training_time.seconds // 3600, self.training_time.seconds // 60 % 60, self.training_time.seconds % 60))
            file.write("Training History: \n %s \n" % pd.DataFrame(self.training_history).to_string(index=False))
            file.write("---------------------------------------------------\n")
            file.write("---------------------------------------------------\n")
            file.write("Run Hyperparameters:\n")

            run_hp = getattr(Model, "run_hparams", None)
            if run_hp is None:
                # fallback (shouldn't happen once build_model sets it)
                run_hp = self._resolve_run_hparams(Model)

            # pretty text in Info.txt
            file.write(json.dumps(run_hp, indent=2, sort_keys=True))
            file.write("\n")
            hparams_json_path = os.path.join(sub_info, "hparams.json")
            run_hp = getattr(Model, "run_hparams", None) or self._resolve_run_hparams(Model)
            with open(hparams_json_path, "w") as jf:
                json.dump(run_hp, jf, indent=2, sort_keys=True)
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
            plt.savefig(os.path.join(sub_plots, f'{Model.model_id}_BinaryCrossEntropy.png'), bbox_inches='tight')

            plt.figure(figsize=(9,6))
            plt.plot(self.training_history['mae'], label='training')
            plt.plot(self.training_history['val_mae'], label='validation')
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend()
            plt.title(Model.model_name+' MAE Loss')
            plt.savefig(os.path.join(sub_plots, f'{Model.model_id}_MAE.png'), bbox_inches='tight')
        else:
            plt.figure(figsize=(9,6))
            plt.plot(self.training_history['loss'], label='training')
            plt.plot(self.training_history['val_loss'], label='validation')
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend()
            plt.title(Model.model_name+' MAE Loss')
            plt.savefig(os.path.join(sub_plots, f'{Model.model_id}_MAE.png'), bbox_inches='tight')

        plt.figure(figsize=(9,6))
        plt.plot(self.training_history['mse'], label='training')
        plt.plot(self.training_history['val_mse'], label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title(Model.model_name+' MSE Loss')
        plt.savefig(os.path.join(sub_plots, f'{Model.model_id}_MSE.png'), bbox_inches='tight')      


    def build_model(self, Model, initial_lr, final_lr_div, lr_power, lr_decay_step):
        """
        Uses instance of ModelBuilder to create a model to train.
        
            Parameters:
                Model (Model object): Model to be built and trained.
                initial_lr (int): Initial learning rate.
                final_lr_div (int): Number by which to divide the initial learning rate to get the final learning rate.
                lr_power (float): Power of the learning rate.
                lr_decay_step (int): Decay step of the learning rate.
                
        """
        
        modelbuilder = ModelBuilder(Model)
        # added optimizer option, works with previous code as it defaults to adam if not specified
        if Model.use_JetPretraining and Model.use_bbPretraining:
            jet_pretrain_model = keras.layers.TFSMLayer(self.jet_pretrain_file, call_endpoints="serving_default")
            bb_pretrain_model = keras.layers.TFSMLayer(self.bb_pretrain_file, call_endpoints="serving_default")
            Model.model = modelbuilder.create_model(initial_lr, final_lr_div, lr_power, lr_decay_step, jet_pretrain_model=jet_pretrain_model, bb_pretrain_model=bb_pretrain_model, optim=self.optimizer)
        elif Model.use_JetPretraining:
            pretrain_model = keras.layers.TFSMLayer(self.jet_pretrain_file, call_endpoints="serving_default")
            Model.model = modelbuilder.create_model(initial_lr, final_lr_div, lr_power, lr_decay_step, jet_pretrain_model=pretrain_model, optim=self.optimizer)
        elif Model.use_bbPretraining:
            bb_pretrain_model = keras.layers.TFSMLayer(self.bb_pretrain_file, call_endpoints="serving_default")
            Model.model = modelbuilder.create_model(
                initial_lr, final_lr_div, lr_power, lr_decay_step,
                bb_pretrain_model=bb_pretrain_model,
                optim=self.optimizer
            )  
        elif Model.unfreeze:
            Model.model = modelbuilder.create_model(initial_lr, final_lr_div, lr_power, lr_decay_step, frozen_file=self.frozen_file, optim=self.optimizer)
        else:
            Model.model = modelbuilder.create_model(initial_lr, final_lr_div, lr_power, lr_decay_step, optim=self.optimizer)
        # get the hparams so i can change architecture
        Model.run_hparams = self._resolve_run_hparams(Model)

        print(Model.model_id+' model has been built and compiled.')
        
        
        
    def train(self, Model):
        """
        Builds, trains, and saves the model.

            Parameters:
                Model (Model object): Model to be built and trained.
        """

        
        # Get the data
        trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = self.load_and_prep(Model)
        print('Data loaded and scaled.')

        Model.hparams = dict(self.arch_hparams)
        # Build the model
        self.build_model(Model, self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step)
        print('Model built.')
            
        # Set early stopping (so no overfitting) and tensorboard callback (for monitoring)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        # use a path utility to get the tensorboard log dir
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_fit_dir(Model.model_id), histogram_freq=0)
        
        # Fit/train the model      
        start = datetime.now()
        history = Model.model.fit([trainX_jets, trainX_other], trainY, verbose=1, epochs=self.max_epochs, validation_data=([valX_jets, valX_other], valY), shuffle=True, callbacks=[early_stop, tensorboard_callback], batch_size=self.batch_size)
        end = datetime.now()
        self.training_time = end - start
        self.training_history = history.history
        print('Training finished.')
        
        # Save the model, its history, and loss plots
        self.save_model(Model)
        print('Model saved.')


    def two_fold_CV(self, template_model, nreps=2, random_state=42):
        '''runs n x 2 cross val on the given model'''
        
        print("##########################################")
        print(f'Starting {nreps} x 2 cross-validation...')
        print("##########################################")

        # load and scale the full dataset
        processor = Utilities()

        self.X_keys, self.Y_keys = processor.getInputKeys(
            template_model.model_v,
            template_model.n_jets,
            template_model.with_ttbar,
            template_model.b_mode
        )

        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(
            self.xmm_file,
            self.ymm_file
        )

        totalX_jets, totalX_other, totalY, \
            self.X_scaled_keys, self.Y_scaled_keys = processor.scale_and_shape(
                self.train_file,
                X_maxmean_dic,
                Y_maxmean_dic,
                self.X_keys,
                self.Y_keys,
                template_model.n_jets,
                template_model.mask_value
            )
        
        N = totalY.shape[0]
        all_idx = np.arange(N, dtype=np.int64)

        # save shapes on the template for reference
        template_model.had_shape = sum(
            'th_' in key or 'wh_' in key for key in self.Y_scaled_keys
        )
        template_model.lep_shape = sum(
            'tl_' in key or 'wl_' in key for key in self.Y_scaled_keys
        )
        template_model.ttbar_shape = sum(
            'ttbar_' in key for key in self.Y_scaled_keys
        )
        template_model.bbbar_shape = sum(
            'b_' in key or 'bbar_' in key or 'b1_' in key or 'b2_' in key
            for key in self.Y_scaled_keys
        )

        print(f"Full dataset loaded and scaled: N = {N}")

        # loop over repetitions and folds

        base_id = template_model.model_id # this is the prefix for the folds
        if not getattr(template_model, "hparams", None):
            template_model.hparams = dict(getattr(self, "arch_hparams", {}) or {})
        else:
            template_model.hparams = dict(template_model.hparams)

        cv_records = [] # store per-fold metrics

        for rep in range(nreps):
            # random permutation of all indexes
            rng = np.random.RandomState(random_state + rep)
            perm = rng.permutation(all_idx)
            # find middle point to split
            mid = N // 2

            idx_A = perm[:mid]
            idx_B = perm[mid:]
            print(f"\n=== Repetition {rep+1}/{nreps} ===")
            print(f"Fold A size: {len(idx_A)}, Fold B size: {len(idx_B)}")

            fold_splits = [(0, idx_A, idx_B),
                           (1, idx_B, idx_A)] # (fold_number, train_idx, val_idx)
            
            for fold_id, train_idx, val_idx in fold_splits:
                print(f"\n--- Training fold {fold_id} (rep {rep}) ---")

                # new model instance for this fold
                Model = TRecNet_Model()

                # copy template params
                Model.initialize(
                    template_model.model_v,
                    template_model.n_jets,
                    template_model.with_ttbar,
                    template_model.b_mode,
                    template_model.use_JetPretraining,
                    template_model.use_bbPretraining,
                    False
                )
                Model.hparams = dict(getattr(template_model, "hparams", {}) or {})
                # unique model ID per fold
                Model.model_id = f"{base_id}_r{rep}_f{fold_id}"

                root_base = os.environ.get("TRECNET_OUTPUT_ROOT", ".")  # default: cwd

                cv_root_dir = os.path.abspath(os.path.join(
                    root_base,
                    "trained_models",
                    template_model.model_v,
                    template_model.model_name,
                    f"{base_id}_CV"
                ))

                fold_dir = os.path.join(cv_root_dir, "folds", f"fold_r{rep}_f{fold_id}")
                
                Model.run_dir_override = fold_dir
                # shapes for this model to use in model saving
                # split from indeces
                trainX_jets = totalX_jets[train_idx]
                trainX_other = totalX_other[train_idx]
                trainY = totalY[train_idx]

                valX_jets = totalX_jets[val_idx]
                valX_other = totalX_other[val_idx]
                valY = totalY[val_idx]

                self.train_size = trainX_jets.shape[0]
                self.val_size = valX_jets.shape[0]

                Model.jets_shape = trainX_jets.shape
                Model.other_shape = trainX_other.shape
                Model.had_shape = template_model.had_shape
                Model.lep_shape = template_model.lep_shape
                Model.ttbar_shape = template_model.ttbar_shape
                Model.bbbar_shape = template_model.bbbar_shape
                Model.mask_value = template_model.mask_value

                # build the keras model
                self.build_model(
                    Model,
                    self.initial_lr,
                    self.final_lr_div,
                    self.lr_power,
                    self.lr_decay_step
                )
                print('model built for this fold.')
                # set early stopping (so no overfitting) and tensorboard callback (for monitoring)
                early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
                # use a path utility to get the tensorboard log dir
                tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_fit_dir(Model.model_id), histogram_freq=0)

                # train
                # train
                fit_kwargs = dict(
                    x=[trainX_jets, trainX_other],
                    y=trainY,
                    verbose=1,
                    epochs=self.max_epochs,
                    validation_data=([valX_jets, valX_other], valY),
                    shuffle=True,
                    callbacks=[early_stop, tensorboard_callback],
                    batch_size=self.batch_size,
                )
                start = datetime.now()
                history = Model.model.fit(**fit_kwargs)
                end = datetime.now()

                self.training_time = end - start
                self.training_history = history.history

                print('Training finished for this fold.')

                self.save_model(Model)
                print('Model saved for this fold.')

                # find the best epoch based on val_loss
                val_losses = np.array(self.training_history["val_loss"])
                best_epoch = int(val_losses.argmin())

                record = {
                    "rep": rep,
                    "fold": fold_id,
                    "model_id": Model.model_id,
                    "best_epoch": best_epoch,
                    "loss": float(self.training_history["loss"][best_epoch]),
                    "val_loss": float(self.training_history["val_loss"][best_epoch]),
                }
                # mse / val_mse
                if "mse" in self.training_history:
                    record["mse"] = float(self.training_history["mse"][best_epoch])
                else:
                    record["mse"] = float("nan")

                if "val_mse" in self.training_history:
                    record["val_mse"] = float(self.training_history["val_mse"][best_epoch])
                else:
                    record["val_mse"] = float("nan")

                cv_records.append(record)

                # save validation preds for this fold into ROOT
                self._save_cv_predictions(
                    Model,
                    base_id,
                    rep,
                    fold_id,
                    val_idx,
                    totalX_jets,
                    totalX_other,
                    totalY,
                    Y_maxmean_dic
                )

        # collect metrics and save to csv/txt
        cv_df = pd.DataFrame(cv_records)
        summary = cv_df[["loss", "val_loss", "mse", "val_mse"]].agg(["mean", "std"])
        # directory to save cv results
        cv_dir = os.path.join(
            "trained_models",
            template_model.model_v,
            template_model.model_name,
            f"{base_id}_CV"
        )
        os.makedirs(cv_dir, exist_ok=True)
        # per-fold table
        cv_csv_path = os.path.join(cv_dir, "cv_folds_metrics.csv")
        cv_df.to_csv(cv_csv_path, index=False)

        # summary text file
        summary_txt_path = os.path.join(cv_dir, "cv_summary.txt")
        with open(summary_txt_path, "w") as f:
            f.write("CV summary (best epoch per fold):\n\n")
            f.write("Per-fold metrics:\n")
            f.write(cv_df.to_string(index=False))
            f.write("\n\nMean/std across all folds:\n\n")
            f.write(summary.to_string())
        print("\nCV metrics saved to:")
        print("  ", cv_csv_path)
        print("  ", summary_txt_path)

        print("\nFinished cross validation.")


    def learning_curve(self, template_model,
                        fractions = (0.05, 0.1, 0.2, 0.3,
                                    0.4, 0.5, 0.6, 0.7, 
                                    0.8, 0.9, 0.99),
                                    nreps=3, 
                                    random_state=42, 
                                    save_models = False, out_tag = None):
        ''' runs a learning curve with the given fractions of data and repetitions,
            using the template model as a base for the runs.
            Saves results in a folder with the template model ID as a prefix and _LC as a suffix,
            with subfolders for each fraction and repetition.
            If save_models is True, saves the trained models for each run (can take a lot of space).
            out_tag can be used to add an additional tag to the output folder name.'''
        
        print("##########################################")
        print(f"Starting learning curve: nreps={nreps}, fracs={fractions}")
        print("##########################################")


        # same set up as for training
        processor = Utilities()

        self.X_keys, self.Y_keys = processor.getInputKeys(template_model.model_v,
                                                          template_model.n_jets,
                                                          template_model.with_ttbar,
                                                          template_model.b_mode)
        
        x_maxmean_dic, y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)

        totalX_jets, totalX_other, totalY, self.X_scaled_keys, self.Y_scaled_keys = processor.scale_and_shape(self.train_file,
                                                                                                              x_maxmean_dic,
                                                                                                              y_maxmean_dic,
                                                                                                              self.X_keys,
                                                                                                              self.Y_keys,
                                                                                                              template_model.n_jets,
                                                                                                              template_model.mask_value)
        template_model.had_shape   = sum('th_' in k or 'wh_' in k for k in self.Y_scaled_keys)
        template_model.lep_shape   = sum('tl_' in k or 'wl_' in k for k in self.Y_scaled_keys)
        template_model.ttbar_shape = sum('ttbar_' in k for k in self.Y_scaled_keys)
        template_model.bbbar_shape = sum(('b_' in k) or ('bbar_' in k) or ('b1_' in k) or ('b2_' in k) for k in self.Y_scaled_keys)
        
        # print(f'Hadronic output shape: {template_model.had_shape}, Leptonic output shape: {template_model.lep_shape}, ttbar output shape: {template_model.ttbar_shape}, bbbar output shape: {template_model.bbbar_shape}')
        # print(f'totalX_jets shape: {totalX_jets.shape}, totalX_other shape: {totalX_other.shape}')

        N = totalY.shape[0]
        all_idx = np.arange(N, dtype=np.int64)
       
        # get indexes for a base train/val split
        base_train_idx, base_val_idx = train_test_split(
            all_idx,
            train_size=self.split,
            shuffle=True,
            random_state=random_state,
        )

        base_id = template_model.model_id
        if not getattr(template_model, "hparams", None):
            template_model.hparams = dict(getattr(self, "arch_hparams", {}) or {})
        else:
            template_model.hparams = dict(template_model.hparams)
        
        lc_dir = os.path.join(
            "trained_models",
            template_model.model_v,
            template_model.model_name,
            f"{base_id}_LC"
        )
        if out_tag:
            lc_dir = os.path.join(lc_dir, out_tag)

        plots_dir = os.path.join(lc_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        records = []

        # helper function to choose random subset of indexes
        def _subsample(idx, frac, rng):
            if frac <= 0:
                raise ValueError("Fraction must be > 0")
            if frac >= 1.0:
                return idx
            m = max(1, int(round(frac*len(idx))))
            m = min(m, len(idx))
            return rng.choice(idx, size=m, replace=False)
        
        # main loops

        for frac in fractions:
            print(f'======Training fraction: {frac}======')
            for rep in range(nreps):
                print(f'--- Rep {rep+1}/{nreps} ---')
                # seed for this instance
                run_seed = int(random_state + 100000 * rep + int(100 * frac))
                # get random generator
                rng = np.random.RandomState(run_seed)
                
                # get train and val idx
                train_idx = _subsample(base_train_idx, frac, rng)
                val_idx = base_val_idx

                # make training tensors
                trainX_jets = totalX_jets[train_idx]
                trainX_other = totalX_other[train_idx]
                trainY = totalY[train_idx]
                # validation tensors
                valX_jets = totalX_jets[val_idx]
                valX_other = totalX_other[val_idx]
                valY = totalY[val_idx]

                # make a model instance for this run
                Model = TRecNet_Model()
                Model.initialize(
                    template_model.model_v,
                    template_model.n_jets,
                    template_model.with_ttbar,
                    template_model.b_mode,
                    template_model.use_JetPretraining,
                    template_model.use_bbPretraining,
                    False
                )
                Model.hparams = dict(getattr(template_model, "hparams", {}) or {})

                # same set up
                Model.model_id = f"{base_id}_lc_frac{int(100*frac):03d}_r{rep+1}"
                Model.jets_shape  = trainX_jets.shape
                Model.other_shape = trainX_other.shape
                Model.had_shape   = template_model.had_shape
                Model.lep_shape   = template_model.lep_shape
                Model.ttbar_shape = template_model.ttbar_shape
                Model.bbbar_shape = template_model.bbbar_shape
                Model.mask_value  = template_model.mask_value
                
                # option to save
                if save_models:
                    run_dir = os.path.join(lc_dir, "runs", Model.model_id)
                    Model.run_dir_override = run_dir
                
                # clear the keras backend to stop memory creep
                tf.keras.backend.clear_session()
                gc.collect()

                self.build_model(Model, self.initial_lr, self.final_lr_div, self.lr_power, self.lr_decay_step)
                print('model built for this run.')
                early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)

                fit_kwargs = dict(
                    x=[trainX_jets, trainX_other],
                    y=trainY,
                    verbose=1,
                    epochs=self.max_epochs,
                    validation_data=([valX_jets, valX_other], valY),
                    shuffle=True,
                    callbacks=[early_stop],
                    batch_size=self.batch_size,
                )

                start = datetime.now()
                history = Model.model.fit(**fit_kwargs)
                end = datetime.now()

                self.training_time = end - start
                self.training_history = history.history

                if save_models:
                    self.save_model(Model)
                    print('Model saved for this run.')

                # find best epoch in validation
                val_losses = np.array(self.training_history.get("val_loss", []), dtype=np.float64)
                best_epoch = int(val_losses.argmin())

                rec = {
                    "frac": float(frac),
                    "rep": int(rep),
                    "seed": int(run_seed),
                    "model_id": Model.model_id,
                    "train_size": int(trainY.shape[0]),
                    "val_size": int(valY.shape[0]),
                    "best_epoch": int(best_epoch),
                    "loss": float(self.training_history["loss"][best_epoch]),
                    "val_loss": float(self.training_history["val_loss"][best_epoch]),
                }

                # include mse if was used
                if "mse" in self.training_history:
                    rec["mse"] = float(self.training_history["mse"][best_epoch])
                if "val_mse" in self.training_history:
                    rec["val_mse"] = float(self.training_history["val_mse"][best_epoch])

                records.append(rec)
                print(f"[LC] frac={frac:.3f} rep={rep} train={rec['train_size']} best_epoch={best_epoch} val_loss={rec['val_loss']:.6g}")

        # save tables

        df = pd.DataFrame(records)
        raw_csv = os.path.join(lc_dir, "learning_curve_raw.csv")
        df.to_csv(raw_csv, index=False)

        # summary mean/std vs train_size
        agg_cols = ["val_loss", "loss"]
        if "val_mse" in df.columns: agg_cols.append("val_mse")
        if "mse" in df.columns: agg_cols.append("mse")

        summary = (
            df.groupby(["frac", "train_size"], as_index=False)[agg_cols]
              .agg(["mean", "std"])
        )

        summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.to_flat_index()]

        summary_csv = os.path.join(lc_dir, "learning_curve_summary.csv")
        summary.to_csv(summary_csv, index=False)
        self._plot_learning_curve(df, plots_dir, metric="loss")
        if "val_mse" in df.columns and "mse" in df.columns:
            self._plot_learning_curve(df, plots_dir, metric="mse")

        print("\nLearning curve saved to:")
        print("  ", raw_csv)
        print("  ", summary_csv)
        print("  ", plots_dir)

        return df, summary
    

    def _plot_learning_curve(self, df, plots_dir, metric="loss"):
        ''' plot the learning curves'''

        train_metric = metric
        val_metric = f"val_{metric}"

        missing = [c for c in (train_metric, val_metric) if c not in df.columns]
        if missing:
            print(f"  [LC] missing columns {missing}, skipping plot for metric='{metric}'.")
            return

        grp = df.groupby("train_size")

        train_mean = grp[train_metric].mean()
        train_std  = grp[train_metric].std().fillna(0.0)

        val_mean   = grp[val_metric].mean()
        val_std    = grp[val_metric].std().fillna(0.0)

        train_sizes = np.array(sorted(train_mean.index.values), dtype=np.int64)

        tm = train_mean.loc[train_sizes].to_numpy(dtype=np.float64)
        ts = train_std.loc[train_sizes].to_numpy(dtype=np.float64)

        vm = val_mean.loc[train_sizes].to_numpy(dtype=np.float64)
        vs = val_std.loc[train_sizes].to_numpy(dtype=np.float64)

        plt.figure(figsize=(9, 6))

        plt.plot(train_sizes, tm, marker='o', linestyle='-', label=f"train {train_metric}")
        plt.fill_between(train_sizes, tm - ts, tm + ts, alpha=0.2)

        plt.plot(train_sizes, vm, marker='o', linestyle='-', label=f"val {val_metric}")
        plt.fill_between(train_sizes, vm - vs, vm + vs, alpha=0.2)

        plt.xlabel("Training set size (events)")
        plt.ylabel(metric)
        plt.title(f"Learning curve: {metric} (train vs val)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        out = os.path.join(plots_dir, f"learning_curve_{metric}.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()


    def _save_cv_predictions(
        self,
        Model,
        base_id,
        rep,
        fold_id,
        val_idx,
        totalX_jets,
        totalX_other,
        totalY,
        Y_maxmean_dic,
    ):
        ''' saves the validation predictions for a fold into a ROOT file,
          with both scaled and original units.'''
        
        # build validation arrays
        valX_jets = totalX_jets[val_idx]
        valX_other = totalX_other[val_idx]
        valY_scaled = totalY[val_idx]  # already scaled

        print(f"Computing validation predictions for model {Model.model_id}...")
        preds_scaled = Model.model.predict(
            [valX_jets, valX_other],
            batch_size=self.batch_size,
            verbose=1,
        )

        # turn into dicts keyed by Y_scaled_keys
        pred_scaled_dic = {}
        true_scaled_dic = {}
        for i, key in enumerate(self.Y_scaled_keys):
            pred_scaled_dic[key] = preds_scaled[:, i]
            true_scaled_dic[key] = valY_scaled[:, i]

        # inverse scale to physical units for output
        scaler = Scaler()
        pred_orig_dic = scaler.invscale_arrays(
            pred_scaled_dic,
            self.Y_scaled_keys,
            Y_maxmean_dic,
        )
        true_orig_dic = scaler.invscale_arrays(
            true_scaled_dic,
            self.Y_scaled_keys,
            Y_maxmean_dic,
        )

        # get only the variables we care about
        # IMPORTANT_TARGETS is defined at the top, should make it a cli argument but later
        keep_keys = [k for k in self.Y_scaled_keys if k in IMPORTANT_TARGETS]
        # make sure the keys we want to keep are actually in the predictions
        # otherwise we get blank root files and waste time
        if not keep_keys:
            raise RuntimeError("No overlap between Y_scaled_keys and IMPORTANT_TARGETS.")

        true_orig_small = {k: true_orig_dic[k] for k in keep_keys}
        pred_orig_small = {k: pred_orig_dic[k] for k in keep_keys}

        # build output directory using the same root_base as two_fold_CV
        root_base = os.environ.get("TRECNET_OUTPUT_ROOT", ".")

        cv_root_dir = os.path.abspath(
            os.path.join(
                root_base,
                "trained_models",
                Model.model_v,
                Model.model_name,
                f"{base_id}_CV",
            )
        )

        fold_dir = os.path.join(cv_root_dir, "folds", f"fold_r{rep}_f{fold_id}")
        results_dir = os.path.join(fold_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        root_path = os.path.join(results_dir, f"Results_r{rep}_f{fold_id}.root")
        print("  Saving validation predictions to:", root_path)

        #  write ROOT file
        with uproot.recreate(root_path) as f:
            f["parton"] = true_orig_small
            f["reco"] = pred_orig_small
            f["parton_scaled"] = {k: true_scaled_dic[k] for k in keep_keys}
            f["reco_scaled"]   = {k: pred_scaled_dic[k] for k in keep_keys}

 
        
    def save_hypertune_results(self,Model,tuner):
        """
        Save the hypertuning results.

            Parameters:
                Model (Model object): Model that we've been hypertuning.
                tuner (Keras tuner): Tuner that we've been hypertuning with.
        """
        
        print('Saving hyperparamter tuning results ...')
        
        dir = os.path.join(
            "trained_models",
            Model.model_v,
            f"{Model.model_name}_hypertune",
            f"{Model.model_id}_Hypertuning"
        )
        # Create directory for saving things in if it doesn't exist
        if not os.path.exists(dir): 
            os.makedirs(dir) 

        
        # Save important information about this model into a text file
        file = open(dir+'/Hypertuning_Info.txt', "w")
        file.write("Model ID: %s \n" % Model.model_id)
        if Model.unfreeze: file.write("Frozen Model ID: %s \n" % self.frozen_model_id)
        if Model.use_JetPretraining: file.write("JetPretrain Model: %s \n" % self.jet_pretrain_file)
        if Model.use_bbPretraining: file.write("bbPretrain Model: %s \n" % self.bb_pretrain_file)
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
        # also record any tuned architecture params
        if getattr(self, "arch_hyper_config", None):
            file.write("\nBest architecture hyperparameters:\n")
            for k in self.arch_hyper_config.keys():
                try:
                    file.write(f"  {k} = {best_hps.get(k)}\n")
                except Exception:
                    pass
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
        # we need to 
        ht_dir = os.path.join(
            "trained_models",
            Model.model_v,
            f"{Model.model_name}_hypertune"
        )
        os.makedirs(ht_dir, exist_ok=True)
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
        tuner.search(x=[trainX_jets, trainX_other], y=trainY, validation_data=([valX_jets, valX_other], valY), epochs=self.max_epochs, callbacks=[early_stop, tensorboard_callback], shuffle=True,verbose=1)
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
        
        def get_arch_hyperparams(self, hp):
            """
            read the arch hyperparam space from the configuration file and make a
              dictionary of architecture hyperparameters.
            """
            cfg = getattr(self.trainer, "arch_hyper_config", {}) or {}
            hp_dic = {}

            for hyperparam, specs in cfg.items():
                if specs["type"] == "choice":
                    hp_dic[hyperparam] = hp.Choice(name=hyperparam, values=specs["choices"])
                elif specs["type"] == "int":
                    hp_dic[hyperparam] = hp.Int(
                        name=hyperparam,
                        min_value=specs["min_value"],
                        max_value=specs["max_value"],
                        step=specs["step"],
                        sampling=specs["sampling"],
                    )
                elif specs["type"] == "float":
                    hp_dic[hyperparam] = hp.Float(
                        name=hyperparam,
                        min_value=specs["min_value"],
                        max_value=specs["max_value"],
                        step=specs["step"],
                        sampling=specs["sampling"],
                    )
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
        
        def _decode_list_hp(self, v):
            """
            turn lists from the tuner into python lists of ints that can be iterated
            over when building the model
            """
            if isinstance(v, str) and ("," in v):
                parts = [p.strip() for p in v.split(",") if p.strip() != ""]
                # ints only for these architecture lists
                return [int(p) for p in parts]
            return v
                    
        def build(self, hp):
            # sample training hparams 
            hp_dic = self.get_hyperparams(hp)

            # sample architecture hparams (optional)
            arch_hp = self.get_arch_hyperparams(hp)

            # need to edit this if you add new hyperparams NOTE TODO
            LIST_KEYS = {
                "jet_td", "bjet_td", "weighted_td", "weighted_b_td",
                "other_mlp", "jet_cls_mlp", "bjet_cls_mlp",
                "lep_head", "had_head", "bb_head",
                "final_mlp",
            }

            for k in list(arch_hp.keys()):
                if k in LIST_KEYS:
                    arch_hp[k] = self._decode_list_hp(arch_hp[k])

            # merge: base config hparams overridden by tuned arch hparams
            base_arch = dict(getattr(self.Model, "hparams", {}) or {})
            base_arch.update(arch_hp)
            self.Model.hparams = base_arch

            # avoid graph / memory buildup across trials
            tf.keras.backend.clear_session()
            gc.collect()

            # build the model with training hparams
            self.trainer.build_model(
                self.Model,
                hp_dic["initial_lr"],
                hp_dic["final_lr_div"],
                hp_dic["lr_power"],
                hp_dic["lr_decay_step"],
            )

            return self.Model.model