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
sys.path.append("/home/jchishol/TRecNet")
sys.path.append("home/jchishol/")
os.environ["CUDA_VISIBLE_DEVICES"]="1"    # These are the GPUs visible for training
from argparse import ArgumentParser
import json

from ModelBuilder import TRecNet_Model
import Training

import tracemalloc
tracemalloc.start()


def pretrain_check(config):
    
    # Initialize some things
    add_jetpretrain = False
    add_bbpretrain = False
    
    # Check that there is a jet pre-train model, with the same number of jets, if needed
    if config["create"]["jet_pretrain"]!=None:
        add_jetpretrain = True
        pretrain_n_jets = int(config["create"]["jet_pretrain"].split('/')[-1].split('jets')[0].split('_')[-1])
        if pretrain_n_jets != config["njets"]:
            print("Please provide a jet pretrain model with the same number of jets as you desire.")
            sys.exit()
        print('Pretrained Jet Classifier added to model.')
        
    # Check that there is a bb pre-train model, with the same number of jets, if needed
    if config["create"]["bb_pretrain"]!=None:
        add_bbpretrain = True
        bb_pretrain_n_jets = int(config["create"]["bb_pretrain"].split('/')[-1].split('jets')[0].split('_')[-1])
        if bb_pretrain_n_jets != config["njets"]:
            print("Please provide a bb pretrain model with the same number of jets as you desire.")
            sys.exit()
        print('Pretrained bb Classifier added to model.')
        
        return add_jetpretrain, add_bbpretrain
    

if __name__ == "__main__":
    
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-v', '--version', help="Architecture version of the model to be trained.", type=str, required=True, choices=['JetPretrainer_v1','TRecNet_tt_v1','TRecNet_ttbb_v1','TRecNet_ttbb_v2','TRecNet_ttbb_v3','TRecNet_ttbb_v4','TRecNet_ttbb_v5'])
    parser.add_argument('-c', '--config_file', help="File (including path) with training (or hypertuning) specifications.", type=str, required=True)
    parser.add_argument('-m', '--mode', help="Whether to create a new model to train, unfreeze an old model, or hypertune a model.", choices=['create','unfreeze','hypertune'])
    
    # Parse the arguments and get the config file
    args = parser.parse_known_args()
    config = json.load(open(args.config_file))
    
    # Create mode
    if args.mode == "create":
        
        print("Starting 'create' training mode.")
        
        # Check if the pretrain files are there and good
        add_jetpretrain, add_bbpretrain = pretrain_check(config)
            
        # Create the model
        Model = TRecNet_Model(args.version, config["njets"], config["create"]["add_ttbar"], add_jetpretrain, add_bbpretrain, False)
        
        # Start the training
        print('Beginning training for '+Model.model_id+'...')
        Trainer = Training(config)
        Trainer.set_create_params(config["create"])
        Trainer.set_train_params(config)
        Trainer.train(Model)
        
        
    # Unfreeze mode
    elif args.mode == "unfreeze":
        
        print("Starting 'unfreeze' training mode.")
        
        # Set some things
        frozen_model_id = config["unfreeze"]["frozen_file"].split('/')[-1]
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
                
        # Create the model
        Model = TRecNet_Model(args.version, config["njets"], add_ttbar, add_jetpretrain, add_bbpretrain, True)
        
        # Start the training
        print('Beginning training for '+Model.model_id+'...')
        Trainer = Training(config)
        Trainer.set_unfreeze_params(config["unfreeze"])
        Trainer.set_train_params(config)
        Trainer.train(Model)
    
    # Hypertune mode
    elif args.mode == "hypertune":
        
        print("Starting 'hypertune' training mode.")
        
        # Check if the pretrain files are there and good
        add_jetpretrain, add_bbpretrain = pretrain_check(config)
        
        # Check that the hypertuner selected is appropriate
        if (config["hypertuning"]["tuner"]!="Hyperband" and config["hypertuning"]["tuner"]!="BayesianOptimization"):
            print('WARNING: Selected tuner is not yet available for TRecNet. Using BayesianOptimization by default.')
            
        # Create the model
        Model = TRecNet_Model(args.version, config["njets"], config["create"]["add_ttbar"], add_jetpretrain, add_bbpretrain, False)
        
        print('Beginning hypertuning for '+Model.model_id+'...')
        Trainer = Training(config)
        Trainer.set_hyper_config(config["hypertuning"])
        Trainer.hypertune(Model)

    print('done :)')