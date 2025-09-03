######################################################################
#                                                                    #
#  MLPrep.py                                                         #
#  Author: Jenna Chisholm                                            #
#  Updated: Jan.11/23                                                #
#                                                                    #
#  For matching ttbar decay products to reco level jets, making h5   #
#  data files, and calculating maxmean of data (all prep necessary   #
#  for TRecNet).                                                     #
#                                                                    #
#  Thoughts for improvements: Use config to get variable names.      #
#                                                                    #
######################################################################


import uproot
import h5py
import numpy as np
import awkward as ak
import pandas as pd
import vector
import sys, os, argparse
from argparse import ArgumentParser
from Util import *

# Define some helpful ranges
# had_range = list(range(1,9))+list(range(-8,0))                     # Quarks have pdgid 1 to 8 (negatives indicate antiparticles)
# lep_range = list(range(11,19))+list(range(-18,10))                 # Leptons have pdgid 11 to 18
# bos_range = list(range(21,26))+list(range(-25,20))+[9,37,-9,-37]   # Bosons have pdgid 21 to 25, and 9, and 37
# light_quarks = list(range(-4,0)) + list(range(1,5))
# b_quarks = [-5,5]



class MLFilePrep:
    """ 
    A class for creating all the necessary files for training or testing the machine learning model.

        Methods:
            getJetsData: Creates an array of dataframes for the jets, including pt, eta, phi, m, and btag for each jet.
            getOtherData: Creates a dataframe for the other (lep, met) variables. Includes lep_pt, lep_eta, lep_phi, met_met, and met_phi.
            getTruthData: Creates a dataframe for the truth variables, including: pt, eta, phi, m, y, E, and pout for thad and tlep; and pt, eta, phi, m, y, E, dphi, Ht, chi, and yboost for ttbar; pt, eta, phi, m for whad and wlep; and isTruth for each of the jets.
            getDataframes: Creates the jet, other, and truth dataframes from a specified file.
            makeH5File: Creates and saves h5 file with the reco and truth variables.
            makeTrainTestH5Files: Creates and saves two h5 files with the reco and truth variables; one for testing and one for training.
            combineH5Files: Combines several h5 files into one file.
    """

    def __init__(self):
        print("Creating filePrepper.")


    def getJetsData(self,reco_tree,keys,var_conf,jn):
        """
        Creates an array of dataframes for the jets, including pt, eta, phi, m, and btag for each jet.

            Parameters:
                reco_tree (root tree): Reco level tree from ROOT file.
                keys (list of str): Keys for the variables in the ROOT file.
                var_conf (str): Name (including path) of the config file for the variable names.
                jn (int): Number of jets you want per event, with padding where need be.

            Returns:
                df_jets (array of dataframes): An array of <jn> dataframes (one for each of the jets), containing jet data.
        """

        # Create an array of <jn> jet dataframes
        df_jets = [pd.DataFrame() for _ in range(jn)]
        
        # Pad pt, eta, phi, e with -2.'s for events with less than <jn> jets, and add these variables to each of the jet dataframes
        for v in ['pt','eta','phi','e']:
            str_jet_var = getObservableName(var_conf,keys,'jet_'+v)
            padded_vars = np.asarray(ak.fill_none(ak.pad_none(reco_tree[str_jet_var], jn, clip=True), -2.))
            for j, df in enumerate(df_jets):
                df[v] = padded_vars[:,j]

        # Pad btag with -2.'s
        str_jet_isbtag = getObservableName(var_conf,keys,'jet_isbtag')
        padded_btags = np.asarray(ak.fill_none(ak.pad_none(reco_tree[str_jet_isbtag], jn, clip=True), -2.))

        # Finish off the dataframes
        for j, df in enumerate(df_jets):

            # Put m in the dataframes and get rid of e (which we won't really need)
            df['m'] = df.apply(lambda row : vector.obj(pt=row['pt'],eta=row['eta'],phi=row['phi'],E=row['e']).mass,axis=1)
            df.drop(columns=['e'],inplace=True)

            # Include btags
            df['isbtag'] = padded_btags[:,j]

        print('Jet dataframes created.')

        return df_jets


    def getOtherData(self,reco_tree,keys,var_conf):
        """
        Creates a dataframe for the other (lep, met) variables. Includes lep_pt, lep_eta, lep_phi, met_met, and met_phi.

            Parameters:
                reco_tree (root tree): Reco level tree from ROOT file
                keys (list of str): Keys for the variables in the ROOT file.
                var_conf (str): Name (including path) of the config file for the variable names.
            
            Returns:
                df_other (dataframe): Dataframe for the other (lep, met) variables.
        """

        # Create dataframe for other variables (note divide lep_pt and met_met by 1000 to convert from MeV to GeV)
        # NOTE: also including jet_n so we can later make a cut on these
        keys = getObservableNamesDict(var_conf,keys,'lep_pt','lep_eta','lep_phi','met_met','met_phi','jet_n')
        df_other = ak.to_dataframe({ML_name:reco_tree[ntuple_name] for ML_name, ntuple_name in keys.items()})

        print('Other dataframe created.')

        return df_other


    def getTruthData(self,truth_tree,keys,var_conf,jn,extra_b_mode,include_jet_truths):
        """
        Creates a dataframe for the truth variables, including: pt, eta, phi, m, y, E, and pout for thad and tlep; and pt, eta, phi, m, y, E, dphi, Ht, chi, and yboost for ttbar; pt, eta, phi, m for whad and wlep; and isTruth for each of the jets.
        
            Parameters: 
                truth_tree (root tree): Parton level tree from ROOT file.
                keys (list of str): Keys for the variables in the ROOT file.
                var_conf (str): Name (including path) of the config file for the variable names.
                jn (int): Number of jets you want per event, with padding where need be.
                extra_b_mode (str): How to include extra b's from ttbb (i.e. b vs bbar, or b1 vs b2).
                include_jet_truths (bool): Flag to include any jet origin truth variables available in the ROOT file.
            
            Returns:
                df_truth (dataframe): Dataframe for the truth variables.
        """

        # Get the truth keys for ttbar, th, tl, wh, and wl
        truth_keys = {}
        for p in ['ttbar_','th_','tl_','wh_','wl_']:
            for v in ['pt','eta','phi','m']:
                truth_keys[p+v] = getObservableName(var_conf,keys,p+v)
                
        # If ttbb, also get b and bbar keys or b1b2 keys
        if extra_b_mode=='bbbar':
            for p in ['b_','bbar_']:
                for v in ['pt', 'eta', 'phi', 'm']:
                    truth_keys[p+v] = getObservableName(var_conf,keys,p+v)
        elif extra_b_mode=='b1b2':
            for p in ['b1_','b2_']:
                for v in ['pt', 'eta', 'phi', 'm']:
                    truth_keys[p+v] = getObservableName(var_conf,keys,p+v)

        # Make the truth table from these keys  
        df_truth = ak.to_dataframe({ML_name: truth_tree[ntuple_name] for ML_name, ntuple_name in truth_keys.items()})

        # Include jet origin info
        if include_jet_truths:
            for jet_truth in ['_isTruth','_isFromttbar','_isExtraB']:
                if jet_truth in keys:
                    padded_matches = np.asarray(ak.fill_none(ak.pad_none(truth_tree['jet'+jet_truth], jn, clip=True), 0))
                    for j in range(jn):
                        df_truth['j'+str(j+1)+jet_truth] = padded_matches[:,j]

        # Include event number (might be useful later)
        #df_truth['eventNumber'] = truth_tree['eventNumber']

        print('Truth dataframe created.')

        return df_truth


    def getDataframes(self,root_file,tree_type,var_conf,jn,extra_b_mode,include_jet_truths):
        """
        Creates the jet, other, and truth dataframes from a specified file.

            Parameters: 
                root_file (str): Name (and path) of the root file you'd like to extract data from.
                tree_type (str): Name of the tree from which to extract the data (nominal, up, or down).
                var_conf (str): Name (including path) of the config file for the variable names.
                jn (int): Number of jets you want per event, with padding where need be.
                extra_b_mode (str): How to include extra b's from ttbb (i.e. b vs bbar, or b1 vs b2).
                include_jet_truths (bool): Flag to include any jet origin truth variables available in the ROOT file.
        
            Returns: 
                df_jets (array of dataframes): An array of <jn> dataframes (one for each of the jets), containing jet data.
                df_other (dataframe): Dataframe for the other (lep, met) variables.
                df_truth (dataframe): Dataframe for the truth variables.
        """

        # Get the desired tree's name
        nom_name, up_name, down_name = getBranchNames(var_conf)
        tree_name = up_name if tree_type=='up' else down_name if tree_type=='down' else nom_name

        # Import the root file data
        root_file = uproot.open(root_file)
        tree = root_file[tree_name].arrays()
        keys = root_file[tree_name].keys()

        # Close root file
        root_file.close()
        
        # Get the array of jet dataframes
        print('Getting jet dataframes ...')
        df_jets = self.getJetsData(tree,keys,var_conf,jn)

        # Get the other reco dataframe
        print('Getting other dataframe ...')
        df_other = self.getOtherData(tree,keys,var_conf)

        # Get the truth dataframe (only for nominal)
        if tree_type=='nominal':
            print('Getting truth dataframe ...')
            df_truth = self.getTruthData(tree,keys,var_conf,jn,extra_b_mode,include_jet_truths)
        else:
            df_truth = ak.to_dataframe({'eventNumber':tree['eventNumber']})
        

        return df_jets, df_other, df_truth


    def makeH5File(self,input,save_dir,tree_type,var_conf,jn,extra_b_mode,include_jet_truths):
        """
        Creates and saves h5 file with the reco and truth variables.

            Parameters: 
                input (str): Name (and path) of the root file you'd like to extract data from.
                save_dir (str): Path for directory where file will be saved.
                tree_type (str): Name of the tree from which to extract the data (nominal, up, or down).
                var_conf (str): Name (including path) of the config file for the variable names.
                jn (int): Number of jets you want per event, with padding where need be.
                extra_b_mode (str): How to include extra b's from ttbb (i.e. b vs bbar, or b1 vs b2).
                include_jet_truths (bool): Flag to include any jet origin truth variables available in the ROOT file.

            Returns: 
                Saves an h5 file with jet, other, and truth data. See getJetData, getOtherData, and getTruthData for details on what's included.
        """
        
        # Separate input file name and its path
        #in_path = os.path.split(self.input_file)[0]
        in_name = os.path.split(input)[1]
        prefix = in_name.split('.root')[0].split('_pruned')[0]

        # Get the data to be saved
        df_jets, df_other, df_truth = self.getDataframes(input,tree_type,var_conf,jn,extra_b_mode,include_jet_truths)

        # Add a tag for the data type and the number of jets included
        tag = '_sysUP' if tree_type=='up' else '_sysDOWN' if tree_type=='down' else '_nom'
        tag = '_bbbar'+tag if extra_b_mode=='bbbar' else '_b1b2'+tag if extra_b_mode=='b1b2' else tag
        tag = '_'+str(jn)+'jets'+tag

        # Creating h5 file for input in the neural network
        new_file_name = save_dir+'/'+prefix+tag+'.h5'
        f = h5py.File(new_file_name,'w')  # "w" means initializing file

        # Create datasets for jets
        for j,df in enumerate(df_jets):
            for v in df.columns:
                f.create_dataset('j'+str(j+1)+'_'+v,data=df[v])

        # Create datasets for other variables
        for v in df_other.columns:
            f.create_dataset(v,data=df_other[v])

        # Data sets for truth variables
        for v in df_truth.columns:        
            f.create_dataset(v,data=df_truth[v])


        print('Saved: '+new_file_name)


    def combineH5Files(self,file_list,output):
        """
        Combines several h5 files into one file.

            Parameters: 
                file_list (list of str): List of the h5 file names (and their paths) you want to combine.
                output (str): Name (and path) the file will be saved to.

            Returns: 
                Saves the combined h5 file (in the same location as the last h5 file in the list).
        """

        # Create a file to combine the data in
        with h5py.File(output+'.h5','w') as h5fw:
            
            current_row = 0   # Keeps track of how many rows of data we have written
            total_len = 0     # Keeps track of how much data we have read
            
            # For each file of that type
            for file in file_list:
                
                # Read the file
                h5fr = h5py.File(file,'r')    # Read the file
                
                # Get the file length and add to the total length of data
                dslen = h5fr['j1_isbtag'].shape[0]
                total_len += dslen

                # For each of the variables
                for key in list(h5fr.keys()):

                    # Get the data
                    arr_data = h5fr[key]
                    
                    # If this is the first data file we're looking at, create the dataset from scratch
                    if current_row == 0:
                        h5fw.create_dataset(key,data=arr_data,maxshape=(None,))
                    # Else, resize the dataset length so that it will fit the new data and then append that data
                    else:
                        h5fw[key].resize((total_len,))
                        h5fw[key][current_row:total_len] = arr_data

                # Update the current row
                current_row = total_len

                print(file+' appended.')

        print('Saved: '+output+'.h5')


    def makeTrainTestH5Files(self,file_list,output,split):
        """
        Combines several h5 files. The first <split>% of events in each file go into a 'train' file, while the remaining events go into a 'test' file.

            Parameters: 
                file_list (list of str): List of the h5 file names (and their paths) you want to combine.
                output (str): Name (and path) the file will be saved to (note: '_train' and '_test' will be appended to the end of this name appropriately).
                split (int or double or float): Percentage of events (expressed as a decimal number) that will go into the training file, while 1-<split> goes to the testing file.

            Returns: 
                Saves the combined h5 file (in the same location as the last h5 file in the list).
        """

        if split<0 or split>1:
            print('Please enter a valid split percentage. Exiting program.')
            sys.exit()

        # Create a file to combine the data in
        if split!=0: h5f_train = h5py.File(output+'_train.h5','w')
        if split!=1: h5f_test = h5py.File(output+'_test.h5','w')
            
        current_train_row = 0   # Keeps track of how many rows of data we have written in the train file
        current_test_row = 0    # Keeps track of how many rows of data we have written in the test file
        total_train_len = 0     # Keeps track of how much data we have read for the train file
        total_test_len = 0      # Keeps track of how much data we have read for the test file
            
        # For each file in the list
        for file in file_list:
                
            # Read the file
            with h5py.File(file,'r') as h5fr:
                
                # Get the file length, split it, and add to the total length of each dataset
                dslen = h5fr['j1_isbtag'].shape[0]
                split_point = int(np.round(dslen*split))

                total_train_len += split_point
                total_test_len += (dslen - split_point)

                # For each of the variables
                for key in list(h5fr.keys()):

                    # As long as it's not all going to testing, append to test data file
                    if split!=0: 
                        train_arr_data = h5fr[key][0:split_point]
                        # If this is the first data file we're looking at, create the dataset from scratch
                        if current_train_row == 0:
                            h5f_train.create_dataset(key,data=train_arr_data,maxshape=(None,))
                        # Else, resize the dataset length so that it will fit the new data and then append that data
                        else:
                            h5f_train[key].resize((total_train_len,))
                            h5f_train[key][current_train_row:total_train_len] = train_arr_data

                    # As long as it's not all going to training, append to test data file
                    if split!=1: 
                        test_arr_data = h5fr[key][split_point:]
                        # If this is the first data file we're looking at, create the dataset from scratch
                        if current_test_row == 0:
                            h5f_test.create_dataset(key,data=test_arr_data,maxshape=(None,))
                        # Else, resize the dataset length so that it will fit the new data and then append that data
                        else:
                            h5f_test[key].resize((total_test_len,))
                            h5f_test[key][current_test_row:total_test_len] = test_arr_data

                # Update the current row
                current_train_row = total_train_len
                current_test_row = total_test_len

            print(file+' data appended.')

        print('Saved: '+output+'_train.h5')
        print('Saved: '+output+'_test.h5')





# ---------- GET ARGUMENTS FROM COMMAND LINE ---------- #

# Create the main parser and subparsers
parser = ArgumentParser()
subparser = parser.add_subparsers(dest='function')
subparser.required = True
p_makeH5File = subparser.add_parser('makeH5File')
p_combineH5Files = subparser.add_parser('combineH5Files')
p_makeTrainTest = subparser.add_parser('makeTrainTestH5Files')

# Define arguments for makeH5File
p_makeH5File.add_argument('--input',help='Input file (including path).',required=True)
p_makeH5File.add_argument('--save_dir',help='Path for directory where file will be saved.',required=True)
p_makeH5File.add_argument('--tree_type',help='Name of the tree from which to extract the data.',choices=['nominal', 'up', 'down'],required=True)
p_makeH5File.add_argument('--var_conf',help='Config file (including path) for names of variables.',required=True)
p_makeH5File.add_argument('--jn',help='Number of jets to include per event (using padding if necessary) (default: 6).',type=int,default=6)
p_makeH5File.add_argument('--extra_b_mode', help='How to include extra bs from ttbb (i.e. b vs bbar, or b1 vs b2).', choices=['none','bbbar','b1b2'],required=True)
p_makeH5File.add_argument('--include_jet_truths', help='Flag to include any jet origin truth variables available in the ROOT file.', action='store_true')

# Define arguments for combineH5Files
p_combineH5Files.add_argument('--file_list',help='Text file containing list of input files (including path).',required=True, type=argparse.FileType('r'))
p_combineH5Files.add_argument('--output',help='Output file (including path).',required=True)

# Define arguments for makeTrainTest
p_makeTrainTest.add_argument('--file_list',help='Text file containing list of input files (including path).',required=True, type=argparse.FileType('r'))
p_makeTrainTest.add_argument('--output',help='Output file (including path).',required=True)
p_makeTrainTest.add_argument('--split',help='Percentage of events (expressed as a decimal number) to include in training file (default: 0.75).',type=float,default=0.75)


# Parse the arguments and proceed with stuff
args = parser.parse_args()
if args.function == 'makeH5File':
    prepper = MLFilePrep()
    prepper.makeH5File(args.input,args.save_dir,args.tree_type,args.var_conf,args.jn,args.extra_b_mode,args.include_jet_truths)
elif args.function == 'combineH5Files':
    file_list = [file.strip() for file in args.file_list]
    prepper = MLFilePrep()
    prepper.combineH5Files(file_list, args.output)
elif args.function == 'makeTrainTestH5Files':
    prepper = MLFilePrep()
    file_list = [file.strip() for file in args.file_list]
    prepper.makeTrainTestH5Files(file_list, args.output, args.split)
else:
    print('Invalid function type.')
    
print('Done :)')