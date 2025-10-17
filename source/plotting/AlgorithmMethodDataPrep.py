######################################################################
#                                                                    #
#  AlgorithmMethodDataPrep.py                                        #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.10/25                                                #
#                                                                    #
#  A class creating "Results.root" files for each of the algorithm   #
#  methods such that they match the format of the ML results.        #
#                                                                    #
######################################################################

# Import useful packages
import argparse
from argparse import ArgumentParser
import uproot
import h5py
import pandas as pd
import awkward as ak
import numpy as np
import vector
import itertools
import tk as tkinter
from Plotter import *
from Particles_and_Observables import PARTICLES



### ----- HELPER FUNCTIONS ----- ###

def getKeys(reco_method,model_keys):
    """
    Get the correct keys for this particular reco method.

        Parameters:
            reco_method (str): Name of the reconstruction method (options: 'KLFitter', 'Chi2', or 'PseudoTop').
            model_keys (list of str): List of the variable keys for this reco method.
            
        Returns:
            truth_names (list of list of str): Combos of the new truth name and the truth name for this model.
            reco_names (list of list of str): Combos of the new reco name and the reco name for this model.
    """
    
    # Get the proper name for this reco method
    model_name = 'PseudoTop_Reco_' if reco_method=='PseudoTop' else 'TtresChi2_' if reco_method=='Chi2' else 'klfitter_bestPerm_'

    # Initialize list of variable names
    reco_names = []
    truth_names = []
    
    # For each particle
    for par in PARTICLES.values():

        # Get all the possible names for this particle
        alt_par_names = list(par.alt_names)
        alt_par_names.append(par.name)

        # For each variable of that particle
        for var in par.observables.values():                  

            # Reset
            found_reco=False
            found_truth=False

            # Get all the possible names for this variable
            alt_var_names = list(var.alt_names)
            alt_var_names.append(var.name)

            # Look through all possible combinations of the alternate names
            for (alt_par_name,alt_var_name) in list(itertools.product(alt_par_names,alt_var_names)):
                if model_name+alt_par_name+'_'+alt_var_name in model_keys:
                    reco_names.append([par.name+'_'+var.name,model_name+alt_par_name+'_'+alt_var_name])
                    found_reco=True
                if model_name+alt_var_name in model_keys: # (particularly, chi_tt)
                    reco_names.append([par.name+'_'+var.name,model_name+alt_var_name])
                    found_reco=True
                if 'MC_'+alt_par_name+'_afterFSR_'+alt_var_name in model_keys:
                    truth_names.append([par.name+'_'+var.name,'MC_'+alt_par_name+'_afterFSR_'+alt_var_name])
                    found_truth=True 
                if 'MC_'+alt_var_name in model_keys:  # (particularly, chi_tt)
                    truth_names.append([par.name+'_'+var.name,'MC_'+alt_var_name])
                    found_truth = True

                # Stop looking for the correct names once we find them
                if found_reco and found_truth:
                    break
                
    return truth_names, reco_names


def createDF(reco_method,filename,truth_keys,reco_keys,test_eventnumbers=[]): 
    """
    Imports data from given file and creates a dataframe.
    
        Parameters:
            reco_method (str): Name of the reconstruction method (options: 'KLFitter', 'Chi2', or 'PseudoTop').
            filename (str): Name of file (including path).
            truth_keys (list of list of str): Combos of the new truth name and the truth name for this model.
            reco_keys (list of list of str): Combos of the new reco name and the reco name for this model.
            
        Optional:
            test_eventnumbers (np.array): List of event numbers that are in the test data set (default: []).
            
        Returns:
            df_nom (pd.DataFrame): Dataframe with nominal data.
            df_up (pd.DataFrame): Dataframe of systematic up data.
            df_down (pd.DataFrame): Dataframe of systematic down data.
    """
	
    # Open root file and its trees
    with uproot.open(filename) as root_file:
        tree_nom = root_file['nominal'].arrays()
        tree_up = root_file['CategoryReduction_JET_Pileup_RhoTopology__1up'].arrays()
        tree_down = root_file['CategoryReduction_JET_Pileup_RhoTopology__1down'].arrays()

    # Did an met_met cut for the ML data, so let's match it here too
    #sel = tree_nom['met_met']/1000 >= 20
    #tree_nom = tree_nom[sel]

    # Also split the datafile the same way we did when making the train/test h5 files
    #split_point = int(np.round(len(tree_nom['eventNumber'])*0.85))
    #tree_nom = tree_nom[split_point:]
    
    # Only take events with the same event numbers as the test data (if desired)
    if len(test_eventnumbers)>0:
        sel = np.isin(tree_nom['eventNumber'],test_eventnumbers)
        tree_nom = tree_nom[sel]

    # Create dataframe(s)!
    df_nom_truth = ak.to_dataframe({new_name:tree_nom[old_name] for (new_name,old_name) in truth_keys})
    df_nom_reco = ak.to_dataframe({new_name:tree_nom[old_name] for (new_name,old_name) in reco_keys})
    df_up = ak.to_dataframe({new_name:tree_up[old_name] for (new_name,old_name) in reco_keys})
    df_down = ak.to_dataframe({new_name:tree_down[old_name] for (new_name,old_name) in reco_keys})

    # Get event numbers
    df_nom_truth['eventNumber'] = tree_nom['eventNumber']
    df_nom_reco['eventNumber'] = tree_nom['eventNumber']
    df_up['eventNumber'] = tree_up['eventNumber']
    df_down['eventNumber'] = tree_down['eventNumber']

    # Include number of jets, so we can look at how they might compare
    df_nom_reco['jet_n'] = tree_nom['jet_n']
    df_up['jet_n'] = tree_up['jet_n']
    df_down['jet_n'] = tree_down['jet_n']

    # Bring in Chi2 for ... Chi2 lol
    if reco_method=='Chi2':
        df_nom_reco['chi2'] = tree_nom['TtresChi2_Chi2']
        df_up['chi2'] = tree_up['TtresChi2_Chi2']
        df_down['chi2'] = tree_down['TtresChi2_Chi2']

        # Cut on eta (had some weird really really large numbers that seem to be reconstruction fails)
        df_nom_reco = df_nom_reco[df_nom_reco['ttbar_eta']<1000]
        df_nom_reco = df_nom_reco[df_nom_reco['ttbar_eta']>-1000]
        df_up = df_up[df_up['ttbar_eta']<1000]
        df_up = df_up[df_up['ttbar_eta']>-1000]
        df_down = df_down[df_down['ttbar_eta']<1000]
        df_down = df_down[df_down['ttbar_eta']>-1000]

    # Include logLikelihood for KLFitter
    elif reco_method=='KLFitter4' or reco_method=='KLFitter6':

        # Note: some events did not have a logLikelihood calculated -- we will need to pad these events to make sure nothing gets shifted weird, and then cut them
        df_nom_reco['logLikelihood'] = ak.flatten(ak.fill_none(ak.pad_none(tree_nom['klfitter_logLikelihood'],1),np.nan))
        df_up['logLikelihood'] = ak.flatten(ak.fill_none(ak.pad_none(tree_up['klfitter_logLikelihood'],1),np.nan))
        df_down['logLikelihood'] = ak.flatten(ak.fill_none(ak.pad_none(tree_down['klfitter_logLikelihood'],1),np.nan))

        # Drop bad klfitter events
        for df in [df_nom_reco, df_up, df_down]:
            df.replace([np.inf,-np.inf],np.nan,inplace=True)
            df.dropna(inplace=True)

    return df_nom_truth, df_nom_reco, df_up, df_down


def appendData(df_nom_truth,df_nom_reco,df_up,df_down,reco_method,filename,truth_keys,reco_keys,test_eventnumbers=[]):
    """
    Appends data from the file to the existing data frames.
    
        Parameters:
            df_nom_truth (pd.DataFrame): Dataframe with nominal truth data.
            df_nom_reco (pd.DataFrame): Dataframe with nominal reco data.
            df_up (pd.DataFrame): Dataframe of systematic up data.
            df_down (pd.DataFrame): Dataframe of systematic down data.
            reco_method (str): Name of the reconstruction method (options: 'KLFitter', 'Chi2', or 'PseudoTop').
            filename (str): Name of file (including path).
            truth_keys (list of list of str): Combos of the new truth name and the truth name for this model.
            reco_keys (list of list of str): Combos of the new reco name and the reco name for this model.
            
        Optional:
            test_eventnumbers (np.array): List of event numbers that are in the test data set (default: []).
        
        Returns:
            df_nom_truth (pd.DataFrame): Dataframe with nominal truth data.
            df_nom_reco (pd.DataFrame): Dataframe with nominal reco data.
            df_up (pd.DataFrame): Dataframe of systematic up data.
            df_down (pd.DataFrame): Dataframe of systematic down data.
    """

    # Get the data from the new file
    if len(test_eventnumbers)>0:
        df_nom_truth_addon, df_nom_reco_addon, df_up_addon, df_down_addon = createDF(reco_method,filename,truth_keys,reco_keys,test_eventnumbers)
    else:
        df_nom_truth_addon, df_nom_reco_addon, df_up_addon, df_down_addon = createDF(reco_method,filename,truth_keys,reco_keys)
        
    # Append data to the main data frame
    df_nom_truth = pd.concat([df_nom_truth,df_nom_truth_addon],axis=0,ignore_index=True)
    df_nom_reco = pd.concat([df_nom_reco,df_nom_reco_addon],axis=0,ignore_index=True)
    df_up = pd.concat([df_up,df_up_addon],axis=0,ignore_index=True)
    df_down = pd.concat([df_down,df_down_addon],axis=0,ignore_index=True)
    
    print('Appended file: '+filename)

    return df_nom_truth,df_nom_reco,df_up,df_down



def makeResultsFile(reco_method,filenames,save_dir,test_file_name=None):
    """
    Creates one results file from a list of filenames for a given reco method.
    
        Parameters:
            reco_method (str): Name of the reconstruction method (options: 'KLFitter', 'Chi2', or 'PseudoTop').
            filenames (list of str): List of file names (including path).
            save_dir (str): Path for directory where file will be saved.
            
        Optional:
            test_file_name (str): Name (including path) of the test data file that was used.
    """
    
    # Get the event numbers, if desired
    if test_file_name!=None:
        with h5py.File(test_file_name,'r') as test_file:
            test_eventnumbers = np.array(test_file.get('eventNumber'))
    else:
        test_eventnumbers = []
    
    # Find the keys first for the reco method
    with uproot.open(filenames[0]) as root_file:
        keys = root_file['nominal'].keys()
    truth_keys, reco_keys = getKeys(reco_method,keys)
    
    # Create the data frames
    nom_truth_df, nom_reco_df, sysUP_df, sysDOWN_df = createDF(reco_method,filenames[0],truth_keys,reco_keys,test_eventnumbers)
    
    # Append data from the other files
    for filename in filenames[1:]:
        nom_truth_df, nom_reco_df, sysUP_df, sysDOWN_df = appendData(nom_truth_df, nom_reco_df, sysUP_df, sysDOWN_df,reco_method,filename,truth_keys,reco_keys,test_eventnumbers)

    # File naming
    if test_file_name!=None:
        new_file_name = reco_method+'_TestDataResults.root'
    else:
        new_file_name = reco_method+'_FullDataResults.root'

    # Save to root file
    f_results = uproot.recreate(save_dir+'/'+new_file_name)
    f_results["parton"] = nom_truth_df
    f_results["reco"] = nom_reco_df
    f_results["sysUP"] = sysUP_df
    f_results["sysDOWN"] = sysDOWN_df
    print('Saved results to :'+save_dir+'/'+new_file_name)
    
    # Close new file
    f_results.close()
    
    



### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
        
    # Create the main parser
    parser = ArgumentParser()

    # Define arguments
    parser.add_argument('--reco_method',help='Reconstruction method name.',required=True,choices=['KLFitter6','KLFitter4','PseudoTop','Chi2'])
    parser.add_argument('--file_list',help='Txt file of file paths.',required=True,type=argparse.FileType('r'))
    parser.add_argument('--save_dir',help='Path for directory where file will be saved.',required=True)
    parser.add_argument('--test_file_name',help='Name (including path) of the test data file.',default=None)

    # Parse the arguments and proceed with stuff
    args = parser.parse_args()
    filelist = [file.strip() for file in args.file_list]
    makeResultsFile(args.reco_method,filelist,args.save_dir,args.test_file_name)

    print('Done :)')