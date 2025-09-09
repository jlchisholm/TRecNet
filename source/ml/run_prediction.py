##########################################################################
#                                                                        #
#  run_TRecNet.py                                                        #
#  Author: Jenna Chisholm                                                #
#  Updated: Sept.8/25                                                    #
#                                                                        #
#  Runs neural network predictions. Can be used for data, systematics,   #
#  or testing the network.                                               # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"    # These are the GPUs visible for training
from argparse import ArgumentParser

from Predictions import Predictor
from TRecNet_Model import TRecNet_Model

# import tracemalloc
# tracemalloc.start()


### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-i', '--model_id', help="ID of the model.", type=str, required=True)
    parser.add_argument('-d', '--test_data', help="Path and file name for the testing data to be used.", type=str, required=True)
    parser.add_argument('-s','--save_loc', help="Directory (including path) in which to save the results.", type=str, required=True)
    parser.add_argument('--testing',help="Flag to say you're testing a TRecNet model. This will save truth values to the results.",type=bool,action="store_true")
    parser.add_argument('--scaled',help="Flag to save the scaled variables in addition to the original variables.",type=bool,action="store_true")

    # Parse arguments
    args = parser.parse_args()
    
    # Load the model
    model = TRecNet_Model()
    model.load(args.model_id)
    
    # Get mode
    mode = 'test' if args.testing else 'data'

    # Test the model
    print('Beginning predicting for '+args.model_id+'...')
    Predictor = Predictor()
    Predictor.predict_and_save_results(model, model, args.test_data, mode, args.save_loc, args.scaled)

    print('Prediction complete! :)')