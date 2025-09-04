######################################################################
#                                                                    #
#  MaxMean.py                                                        #
#  Author: Jenna Chisholm                                            #
#  Updated: Jun.5/25                                                 #
#                                                                    #
#  For creating maxmean files for training TRecNet.                  #
#                                                                    #
#  Thoughts for improvements:                                        #
#                                                                    #
######################################################################

import h5py
import numpy as np
import pandas as pd
from argparse import ArgumentParser

    
class MaxMeanMachine:
    """
    A class for calculating and saving a dictionary of [max,mean] values for X (reco) and Y (truth) variables.
    
        Methods:
            saveMaxMean: Saves numpy array of [max,mean] values for X (reco) and Y (truth) variables.
    """

    def __init__(self):
        print("Creating MaxMeanMachine.")
    
    def saveMaxMean(self,input_file,save_dir,extra_b_mode):
        """
        Saves dictionary of [max,mean] values for X (reco) and Y (truth) variables.
        
            Parameters: 
                input_file (str): Name (and path) of the h5 file you want to calculate the max mean values for.
                save_dir (str): Directory where you would like to save the max mean values.
                extra_b_mode (str): How to include extra b's from ttbb (i.e. b vs bbar, or b1 vs b2).

            Returns: 
                Saves two numpy files of [max, mean] values; one for X (reco) variables and one for Y (truth) variables.
        """

        print('Opening file ...')

        # Load data
        f = h5py.File(input_file,'r')    

        print('File opened.')

        name = input_file.split('/')[-1].split('.h5')[0]

        # Create data frame
        df = pd.DataFrame({key: np.array(f.get(key)) for key in list(f.keys())})

        # Initialize dictionaries of maxmean
        X_maxmean = {}
        Y_maxmean = {}

        # Get the number of jets per event
        jn = len(list(filter(lambda a: 'j' in a and 'pt' in a, f.keys())))
        f.close()

        # Jets
        for j in range(jn):

            # Calculate px and py
            df['j'+str(j+1)+'_px'] = df['j'+str(j+1)+'_pt']*np.cos(df['j'+str(j+1)+'_phi'])
            df['j'+str(j+1)+'_py'] = df['j'+str(j+1)+'_pt']*np.sin(df['j'+str(j+1)+'_phi'])
    
            # Append max and mean
            for v in ['pt','px','py','eta','m','isbtag']:
                X_maxmean['j'+str(j+1)+'_'+v] = [df['j'+str(j+1)+'_'+v].abs().max(),df['j'+str(j+1)+'_'+v].mean()]
            
            # Also append isTruth
            for v in ['_isTruth','_isFromttbar','_isExtraB']:
                Y_maxmean['j'+str(j+1)+v] = [df['j'+str(j+1)+v].abs().max(),df['j'+str(j+1)+v].mean()]

        print('Jets done.')

        # Calculate px and py of lep
        df['lep_px'] = df['lep_pt']*np.cos(df['lep_phi'])
        df['lep_py'] = df['lep_pt']*np.sin(df['lep_phi'])

        # Calculate sin(met_phi) and cos(met_phi)
        df['met_phi-sin'] = np.sin(df['met_phi'])
        df['met_phi-cos'] = np.cos(df['met_phi'])

        # Append maxmean for other variables
        for v in ['lep_pt','lep_px','lep_py','lep_eta','met_met','met_phi-sin','met_phi-cos']:
            X_maxmean[v] = [df[v].abs().max(),df[v].mean()]

        print('Other done.')

        # Save array of X maxmean values
        np.save(save_dir+'/X_maxmean_'+name,X_maxmean)
        print('Saved: '+save_dir+'/X_maxmean_'+name+'.npy')
        print("X_maxmean keys:")
        print(X_maxmean.keys())

        # Calculate px and py for truth
        particles = ['th_','wh_','tl_','wl_','ttbar_']
        if extra_b_mode == 'bbbar':
            particles.extend(['b_','bbar_'])
        elif extra_b_mode == 'b1b2':
            particles.extend(['b1_','b2_'])
            
        for p in particles:
            df[p+'px'] = df[p+'pt']*np.cos(df[p+'phi'])
            df[p+'py'] = df[p+'pt']*np.sin(df[p+'phi'])

            # Append maxmean for all truth variables
            for v in ['pt','px','py','eta','m']:
                Y_maxmean[p+v] = [df[p+v].abs().max(),df[p+v].mean()]
                
        print('Truth done.')

        # Save Y maxmean arrays
        np.save(save_dir+'/Y_maxmean_'+name,Y_maxmean)
        print('Saved: '+save_dir+'/Y_maxmean_'+name+'.npy')
        print("Y_maxmean keys:")
        print(Y_maxmean.keys())
        
  
# ---------- GET ARGUMENTS FROM COMMAND LINE ---------- #      
        
# Create the main parser
parser = ArgumentParser()

# Define arguments for saveMaxMean
parser.add_argument('--input',help='Input training h5 file (including path).',required=True)
parser.add_argument('--save_dir',help='Path for directory where file will be saved.',required=True)
parser.add_argument('--extra_b_mode', help='How to include extra bs from ttbb (i.e. b vs bbar, or b1 vs b2).', choices=['none','bbbar','b1b2'],required=True)

# Parse the arguments and proceed with stuff
args = parser.parse_args()
mm = MaxMeanMachine()
mm.saveMaxMean(args.input,args.save_dir,args.extra_b_mode)

print('Done :)')