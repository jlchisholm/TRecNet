#########################################################################
#                                                                       #
#  InfoGrabber.py.                                                      #
#  Author: Jenna Chisholm                                               #
#  Updated: Sept.8/25                                                   #
#                                                                       #
#  Defines classes and functions to help get info from trained models.  # 
#                                                                       #
#  Thoughts for improvements: Have X and Y keys as input variables?     #
#                                                                       #
#########################################################################

import sys, os

class InfoGrabber:
    """
    A class for using pre-existing neural networks.
    """  
    
    def __init__(self):
        """
        Initializes an InfoGrabber object. 
        """

        
    def check_model_trained(self, model_id):
        """
        Checks that the input model id actually corresponds to a trained model.
        
            Parameters:
                model_id (str): ID of the trained model.
        """
        
        trained_models_list = os.listdir('trained_models/')
        if (model_id not in trained_models_list):
            print('There is not trained model with this ID. Exiting program.')
            sys.exit()
        

    def get_train_data_file(self, model_id):
        """
        Extracts the training data h5 file name from the training info.

            Parameters:
                model_id (str): ID of the trained model.
                
            Returns:
                train_data_file (str): Name (including path) of the h5 train data file.
        """
        
        train_data_file = None
        info_file = 'trained_models/'+model_id+'/'+model_id+'_Info.txt'
        with open(info_file) as file:
            for line in file:
                if 'Training Data File: ' in line:
                    train_data_file = line.split('Training Data File: ')[1]
                    break
                    
        if train_data_file == None:
            print('Failed to find train data. Program exiting.')
            sys.exit()
                
        return train_data_file
    
    
    def get_maxmean_files(self, model_id):
        """
        Extracts the maxmean file names from the training info.

            Parameters:
                model_id (str): ID of the trained model.
                
            Returns:
                xmm_file (str): Name (including path) of the xmaxmean file. 
                ymm_file (str): Name (including path) of the ymaxmean file.           
        """
        
        model_file_list = os.listdir('trained_models/'+model_id+'/')
        xmm_file = next((f for f in model_file_list if 'X_maxmean' in f), None)
        ymm_file = next((f for f in model_file_list if 'Y_maxmean' in f), None)
        if (xmm_file==None or ymm_file==None):
            print('Failed to find maxmean files. Something must be terribly wrong. Exiting program.')
            sys.exit()
            
        return xmm_file, ymm_file
    
    def get_train_val_split(self, model_id):
        """
        Extracts the training data h5 file name from the training info.

            Parameters:
                model_id (str): ID of the trained model.
                
            Returns:
                split (float): Percentage of data from training file that will be given to training, while the remainder is used for validation (taken from config file).
        """
        
        split = None
        info_file = 'trained_models/'+model_id+'/'+model_id+'_Info.txt'
        with open(info_file) as file:
            for line in file:
                if 'Percentage of Train Data Used for Training: ' in line:
                    split = line.split('Percentage of Train Data Used for Training: ')[1]
                    break
                    
        if split == None:
            print('Failed to find train data. Program exiting.')
            sys.exit()
                
        return split
    
    
    def get_data_type(self, data_file):
        """
        Extracts the data type ('nom', 'sysUP', or 'sysDOWN' from the data file name).
        
            Parameters:
                data_file (str): Name (including path) of the data file.
            Returns:
                data_type (str): Type of data for the data file ('nom', 'sysUP', or 'sysDOWN').
        """
            
        if 'nom' in data_file:
            data_type = 'nom' 
        elif 'sysUP' in data_file:
            data_type = 'sysUP'
        elif 'sysDOWN' in data_file:
            data_type = 'sysDOWN'
        else:
            print('Failed to figure out data type from file name. Exiting program.')
            sys.exit()
            
        return data_type
    
    def get_model_v(self, model_id):
        
        model_v = model_id.split('v')[0] + 'v' + model_id.split('v')[1].split('_')[0]
        return model_v
    
    def get_njets(self, model_id):
        
        n_jets = model_id.split('jets')[0].split('_')[-1]
        return n_jets
    
    def get_ttbar_status(self, model_id):
        
        add_ttbar = True if '+ttbar' in model_id else False
        return add_ttbar
    
    def get_b_mode(self, model_id):
        
        b_mode = 'bbbar' if 'bbbar' in model_id else 'b1b2' if 'b1b2' in model_id else None
        return b_mode