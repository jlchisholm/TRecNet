##########################################################################
#                                                                        #
#  run_learning_curve.py                                                 #
#                                                                        #
#  Train many models across train-set sizes and plot learning curves.     #
#                                                                        #
##########################################################################

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from argparse import ArgumentParser
import json

from TRecNet_Model import TRecNet_Model
from Training import Training

# Same compatibility dict in run_training.py
compatible_models = {
    "JetClassifier_v1": [
        "TRecNet_tt_v1", "TRecNet_ttbb_v1", "TRecNet_ttbb_v2", "TRecNet_ttbb_v3",
        "TRecNet_ttbb_v4", "TRecNet_ttbb_v5", "TRecNet_ttbb_v4x0", "TRecNet_ttbb_v5x0"
    ],
    "bbClassifier_v1": ["TRecNet_ttbb_v1"],
}

def pretrained_classifier_check(model_version, config, classifier):
    '''    Check if a pretrained classifier is provided and compatible with the TRecNet model.
    If so, return True to add it to the model, otherwise return False.'''

    # check that there is a jet pre-train model
    if config["create"][classifier] is not None:
        add_pretrained_classifier = True

        # ensure number of jets is okay
        classifier_n_jets = int(config["create"][classifier].split('/')[-1].split('jets')[0].split('_')[-1])
        if classifier_n_jets != config["njets"]:
            print("Please provide a " + classifier + " model with the same number of jets as you desire.")
            sys.exit()
        # ensure compatibility with TRecNet model version
        classifier_id = config["create"][classifier].split('/')[-1]
        classifier_version = classifier + classifier_id.split('_')[1]
        if model_version not in compatible_models.get(classifier_version, []):
            print("Pretrained classifier version " + classifier_version + " is not compatible with " + model_version + "!")
            sys.exit()

        print('Pretrained classifier added to model.')
    else:
        add_pretrained_classifier = False

    return add_pretrained_classifier

def parse_fracs(s):
    '''Parse a comma-separated string of fractions into a tuple of floats.'''

    # Remove whitespace and split by comma
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # Convert to floats and validate that all are > 0
    fracs = tuple(float(p) for p in parts)
    for f in fracs:
        if f <= 0:
            raise ValueError("All fractions must be > 0")
    return fracs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-v', '--version',
        help="Architecture version of the model to be trained.",
        type=str,
        required=True,
        choices=[
            'JetClassifier_v1', 'bbClassifier_v1',
            'TRecNet_tt_v1', 'TRecNet_ttbb_v1', 'TRecNet_ttbb_v2', 'TRecNet_ttbb_v3',
            'TRecNet_ttbb_v4', 'TRecNet_ttbb_v5',
            'TRecNet_ttbb_v4x0', 'TRecNet_ttbb_v5x0',
            'TRecNet_ttbb_v5x1', 'TRecNet_ttbb_v5x1_clf',
            'TRecNet_ttbb_v5x2', 'TRecNet_ttbb_v5x3'
        ]
    )
    parser.add_argument('-c', '--config_file', type=str, required=True)
    parser.add_argument('--fracs', type=str, default="0.05,0.1,0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated train fractions of base train split.")
    parser.add_argument('--nreps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_models', action='store_true',
                        help="If set, saves each trained model under the learning-curve directory.")
    parser.add_argument('--tag', type=str, default=None,
                        help="Optional subfolder tag under <model_id>_LC/")

    args = parser.parse_args()
    config = json.load(open(args.config_file))

    # check pretrainers like run_training create-mode
    add_jetpretrain = pretrained_classifier_check(args.version, config, "pretrained_jet_classifier")
    add_bbpretrain  = pretrained_classifier_check(args.version, config, "pretrained_bb_classifier")

    # build template model to carry config/ID we need
    Model = TRecNet_Model()
    Model.initialize(args.version, config["njets"], config["create"]["add_ttbar"], config["create"]["b_mode"],
                     add_jetpretrain, add_bbpretrain, False)

    Trainer = Training(config)
    Trainer.set_create_params(config["create"])
    Trainer.set_train_params(config)

    fracs = parse_fracs(args.fracs)

    print(f"Running learning curve for {Model.model_id}")
    Trainer.learning_curve(
        template_model=Model,
        fractions=fracs,
        nreps=args.nreps,
        random_state=args.seed,
        save_models=args.save_models,
        out_tag=args.tag,
    )

    print("done :)")