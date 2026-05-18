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

from argparse import ArgumentParser
# import tracemalloc
# tracemalloc.start()

# All imports relative to TRecNet/source/ directory
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1") 
from source.ml.Predictions import Predictor
from source.ml.TRecNet_Model import TRecNet_Model
from source.ml.Models.blocks import set_encoder, transformer_blocks, objwise, pooling


### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-i', '--model_id', help="ID of the model.", type=str, required=True)
    parser.add_argument('-d', '--test_data', help="Path and file name for the testing data to be used.", type=str, required=True)
    parser.add_argument('-s','--save_loc', help="Directory (including path) in which to save the results.", type=str, required=True)
    # got rid of type=bool as it was throwing an error
    parser.add_argument('--testing',help="Flag to say you're testing a TRecNet model. This will save truth values to the results.",action="store_true")
    parser.add_argument('--include_scaled',help="Flag to save the scaled variables in addition to the original variables.",action="store_true")

    # Parse arguments
    args = parser.parse_args()
    
    # Load the model
    model = TRecNet_Model()
    model.load(args.model_id)
    
    # Get mode
    mode = 'test' if args.testing else 'data'

    # Test the model
    print('Beginning predicting for '+args.model_id+'...')

    #put predictor in lower case as it was confusing it with the Predictor class
    predictor = Predictor()
    predictor.predict_and_save_results(model, args.test_data, mode, args.save_loc, args.include_scaled)

    print('Prediction complete! :)')