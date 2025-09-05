##########################################################################
#                                                                        #
#  run_training.py                                                       #
#  Author: Jenna Chisholm                                                #
#  Updated: Sept.5/25                                                    #
#                                                                        #
#  Runs neural network testing.                                          # 
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
from Testing import Testing

import tracemalloc
tracemalloc.start()



### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-i', '--model_id', help="ID of the model.", type=str, required=True)
    parser.add_argument('-d', '--test_data', help="Path and file name for the testing data to be used.", type=str, required=True)
    parser.add_argument('--data_type', choices=['nominal','sysUP','sysDOWN'],help='Type of data.', required=True)
    parser.add_argument('--save_loc', help="Directory (including path) in which to save the results.", type=str, required=True)
    #parser.add_argument('--ttbb', help='to add b and bbar testing', action="store_true")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check that such a trained model exists
    trained_models_list = os.listdir('trained_models/')
    if (args.model_id not in trained_models_list):
        print('There is not trained model with this ID. Exiting program.')
        sys.exit()
      
    # Check that we're not using the same data file trained on  
    info_file = 'trained_models/'+args.model_id+'/'+args.model_id+'_Info.txt'
    with open(info_file) as file:
        for line in file:
            if 'Training Data File: ' in line:
                train_data = line.split('Training Data File: ')[1]
                if train_data == args.test_data:
                    print('WARNING: You are testing with the same data this model was trained on!!!')
        
    # Grab xmaxmean and ymaxmean file names
    model_file_list = os.listdir('trained_models/'+args.model_id+'/')
    xmm_file = next((f for f in model_file_list if 'X_maxmean' in f), None)
    ymm_file = next((f for f in model_file_list if 'Y_maxmean' in f), None)
    if (xmm_file==None or ymm_file==None):
        print('Failed to find maxmean files. Something must be terribly wrong. Exiting program.')
        sys.exit()
    
    # Test the model
    print('Beginning testing for '+args.model_id+'...')
    
    Tester = Testing(args.test_data, xmm_file, ymm_file, args.data_type)
    Tester.test(args.model_id,args.save_loc)
    
    