##########################################################################
#                                                                        #
#  run_training.py                                                       #
#  Author: Jenna Chisholm                                                #
#  Updated: Jul.23/25                                                    #
#                                                                        #
#  Runs neural network training.                                         # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################


import os, sys
# makes intra-repo imports work regardless of CWD
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.environ["CUDA_VISIBLE_DEVICES"]="1"    # These are the GPUs visible for training
from argparse import ArgumentParser
import json

from TRecNet_Model import TRecNet_Model
from Training import Training

import tracemalloc
tracemalloc.start()

# Dictionary of compatible models for each of the pretrained classifiers
compatible_models = {
    "JetClassifier_v1": [
        "TRecNet_tt_v1",
        "TRecNet_ttbb_v1",
        "TRecNet_ttbb_v2",
        "TRecNet_ttbb_v3",
        "TRecNet_ttbb_v4",
        "TRecNet_ttbb_v5",
        "TRecNet_ttbb_v4x0",
        "TRecNet_ttbb_v5x0",
        "TRecNet_ttbb_v5x1",
        "TRecNet_ttbb_v5x1_clf",
        "TRecNet_ttbb_v5x2",
        "TRecNet_ttbb_v5x3",
    ],
    "bbClassifier_v1": ["TRecNet_ttbb_v1", "TRecNet_ttbb_v5"],
}


def pretrained_classifier_check(model_version, config, classifier):
    """
    Saves the model itself, the training history, and plots of the training loss.

        Parameters:
            model_version (str): Architecture version of the model to be trained (e.g. 'TRecNet_ttbb_v2').
            config (dictionary): Dictionary of config settings for the training.
            classifier (str): Classifier to check ('pretrained_jet_classifier' or 'pretrained_bb_classifier')

        Returns:
            add_pretrainer_classifier (bool): Whether or not this classifier should be added. Returns False if the config file doesn't list a pretrained classifer and True if it does (and this pretrained model is compatible with other settings.)
            
    """
    
    # Check that there is a jet pre-train model
    if config["create"][classifier]!=None:
        add_pretrained_classifier = True
        
        # Ensure number of jets is okay
        classifier_n_jets = int(config["create"][classifier].split('/')[-1].split('jets')[0].split('_')[-1])
        if classifier_n_jets != config["njets"]:
            print("Please provide a "+classifier+" model with the same number of jets as you desire.")
            sys.exit()
            
        # Check that pretrained classifier model is compatible with TRecNet model
        classifier_id = config["create"][classifier].split('/')[-1]
        classifier_version = classifier+classifier_id.split('_')[1]
        if model_version not in compatible_models.get(classifier_version, []):
            print("Pretrained classifier version "+ classifier_version + " is not compatible with " + model_version + "!")
            sys.exit()
            
        print('Pretrained classifier added to model.')
        
    else:
        add_pretrained_classifier = False
            
    return add_pretrained_classifier


### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-v', '--version', help="Architecture version of the model to be trained.", type=str, required=True, choices=['JetClassifier_v1', 'bbClassifier_v1', 'TRecNet_tt_v1','TRecNet_ttbb_v1','TRecNet_ttbb_v2','TRecNet_ttbb_v3','TRecNet_ttbb_v4','TRecNet_ttbb_v5', 'TRecNet_ttbb_v4x0', 'TRecNet_ttbb_v5x0', 'TRecNet_ttbb_v5x1','TRecNet_ttbb_v5x1_clf', 'TRecNet_ttbb_v5x2', 'TRecNet_ttbb_v5x3'])
    parser.add_argument('-c', '--config_file', help="File (including path) with training (or hypertuning) specifications.", type=str, required=True)
    parser.add_argument('-m', '--mode', help="Whether to create a new model to train, unfreeze an old model, or hypertune a model.", choices=['create','unfreeze','hypertune'])
    
    # Parse the arguments and get the config file
    args = parser.parse_args()
    config = json.load(open(args.config_file))
    
    # Create mode
    if args.mode == "create":
        
        print('===================================')
        print('=           CREATE MODE           =')
        print('===================================')
         
        # Check if the pretrain files are there and good
        add_jetpretrain = pretrained_classifier_check(args.version, config, "pretrained_jet_classifier")
        add_bbpretrain = pretrained_classifier_check(args.version, config, "pretrained_bb_classifier")
            
        # Create the model
        Model = TRecNet_Model()
        Model.initialize(args.version, 
                         config["njets"], 
                         config["create"]["add_ttbar"], 
                         config["create"]["b_mode"],
                         add_jetpretrain, 
                         add_bbpretrain,
                         False)
        # this works to get the hparams in the model, but kinda eh
        Model.hparams = config.get("hparams", {}) or {}
        # Start the training
        print('Beginning training for '+Model.model_id+'...')
        Trainer = Training(config)
        Trainer.set_create_params(config["create"])
        Trainer.set_train_params(config)
        Trainer.train(Model)
        
        
    # Unfreeze mode
    elif args.mode == "unfreeze":
        
        print('===================================')
        print('=          UNFREEZE MODE          =')
        print('===================================')
        
        # Set some things
        # extracts b_mode, add_ttbar, and booleans for +JetPretrain and +bbPretrain from the frozen model's ID
        # was having trouble getting these from the config file directly
        # not a good long-term solution, but works for now
        frozen_model_id = config["unfreeze"]["frozen_file"].split('/')[-1]
        b_mode = 'bbbar' if 'bbbar' in frozen_model_id else 'b1b2' if 'b1b2' in frozen_model_id else None
        add_ttbar = True if '+ttbar' in frozen_model_id else False
        add_jetpretrain = True if '+JetPretrain' in frozen_model_id else False
        add_bbpretrain = True if '+bbPretrain' in frozen_model_id else False
        
        # Check that there is a frozen version of the model, with the same number of jets, if needed
        if config["unfreeze"]["frozen_model"]==None:
            print('Please provide a frozen version of the model in order to unfreeze and fine-tune weights.')
            sys.exit()
        else:
            frozen_n_jets = int(config["unfreeze"]["frozen_model"].split('/')[-1].split('jets')[0].split('_')[-1])
            if frozen_n_jets != config["njets"]:
                print("Please provide a frozen model with the same number of jets as you desire.")
                sys.exit()
                
        # Create the model.  # this should be load I feel like, change later!
        Model = TRecNet_Model()
        Model.initialize(args.version, config["njets"], add_ttbar, b_mode, add_jetpretrain, add_bbpretrain, True)
        # meh way to get the hparams in the model, but it works for now
        Model.hparams = config.get("hparams", {}) or {}
        # Start the training
        print('Beginning training for '+Model.model_id+'...')
        Trainer = Training(config)
        Trainer.set_unfreeze_params(config["unfreeze"], Model)
        Trainer.set_train_params(config)
        Trainer.train(Model)
    
    # Hypertune mode
    elif args.mode == "hypertune":

        print('====================================')
        print('=          HYPERTUNE MODE          =')
        print('====================================')

        # Check if the pretrain files are there and good
        add_jetpretrain = pretrained_classifier_check(args.version, config, "pretrained_jet_classifier")
        add_bbpretrain  = pretrained_classifier_check(args.version, config, "pretrained_bb_classifier")

        # accept either "hypertune" or legacy "hypertuning" cause I have models that take
        # architecture hyperparameters that I want to hypertune
        ht_cfg = config.get("hypertune", None)
        if ht_cfg is None:
            ht_cfg = config.get("hypertuning", None)
        if ht_cfg is None:
            raise KeyError("Config must contain 'hypertune' (or legacy 'hypertuning') block.")

        if ht_cfg.get("tuner") not in ("Hyperband", "BayesianOptimization"):
            print("WARNING: tuner must be 'Hyperband' or 'BayesianOptimization'. Defaulting to BayesianOptimization.")
            ht_cfg["tuner"] = "BayesianOptimization"

        Model = TRecNet_Model()
        Model.initialize(
            args.version,
            config["njets"],
            config["create"]["add_ttbar"],
            config["create"]["b_mode"],
            add_jetpretrain,
            add_bbpretrain,
            False
        )

        # baseline arch params (can be overridden by arch_hyperparams per trial)
        Model.hparams = config.get("hparams", {}) or {}

        print('Beginning hypertuning for ' + Model.model_id + '...')
        Trainer = Training(config)

        # needed for hypertune build_model()
        Trainer.set_create_params(config["create"])
        Trainer.set_train_params(config)

        Trainer.set_hyper_config(ht_cfg)
        Trainer.hypertune(Model)

    print('done :)')
