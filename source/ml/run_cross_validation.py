
############
# run_cross_validation.py
# perform nx2 cross validation on a model
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from argparse import ArgumentParser
import json
from TRecNet_Model import TRecNet_Model
from Training import Training

compatible_models = {
    "JetClassifier_v1": [
        "TRecNet_tt_v1", "TRecNet_ttbb_v1", "TRecNet_ttbb_v2",
        "TRecNet_ttbb_v3", "TRecNet_ttbb_v4", "TRecNet_ttbb_v5",
        "TRecNet_ttbb_v4x0", "TRecNet_ttbb_v5x0",
        "TRecNet_ttbb_v5x1", "TRecNet_ttbb_v5x1_clf",
        "TRecNet_ttbb_v5x2", "TRecNet_ttbb_v5x3"
    ],
    "bbClassifier_v1": ["TRecNet_ttbb_v1", "TRecNet_ttbb_v5"],
}

def pretrained_classifier_check(model_version, config, classifier):
    """
    Check if a pretrained classifier is provided and compatible with the TRecNet model.
    If so, return True to add it to the model, otherwise return False.
    """
    
    # check that there is a jet pre-train model
    if config["create"][classifier]!=None:
        add_pretrained_classifier = True
        
        # ensure number of jets is okay
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

def main():
    parser = ArgumentParser()
    # input parameters
    parser.add_argument('-v',
                        '--version',
                        help="Architecture version of the model to be trained.",
                        type=str, required=True,
                        choices=['JetClassifier_v1',
                                'bbClassifier_v1',
                                'TRecNet_tt_v1',
                                'TRecNet_ttbb_v1',
                                'TRecNet_ttbb_v2',
                                'TRecNet_ttbb_v3',
                                'TRecNet_ttbb_v4',
                                'TRecNet_ttbb_v5',
                                'TRecNet_ttbb_v4x0',
                                'TRecNet_ttbb_v5x0',
                                'TRecNet_ttbb_v5x1',
                                'TRecNet_ttbb_v5x1_clf',
                                'TRecNet_ttbb_v5x2',
                                'TRecNet_ttbb_v5x3'])
    
    parser.add_argument('-d', '--data', required=True)

    parser.add_argument('-n', '--nreps',type=int, default = 2)

    parser.add_argument('-c', '--config_file',
                        type=str, required=True)

    # get arguments
    args = parser.parse_args()
    # get config from file using parse args
    config = json.load(open(args.config_file))
    config["data"] = args.data

    add_jetpretrain = pretrained_classifier_check(args.version, config, "pretrained_jet_classifier")
    add_bbpretrain = pretrained_classifier_check(args.version, config, "pretrained_bb_classifier")
    

    # templeate model to derive per fold model IDs
    Model = TRecNet_Model()

    Model.initialize(args.version, 
                     config["njets"], 
                     config["create"]["add_ttbar"], 
                     config["create"]["b_mode"],
                     add_jetpretrain, 
                     add_bbpretrain, 
                     unfreeze_mode=False)

    Trainer = Training(config)
    Trainer.set_create_params(config["create"])
    Trainer.set_train_params(config)

    Trainer.two_fold_CV(Model, nreps=args.nreps, random_state = 42)




if __name__ == '__main__':
    main()
