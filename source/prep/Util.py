######################################################################
#                                                                    #
#  Util.py                                                           #
#  Author: Jenna Chisholm                                            #
#  Updated: Jun.5/25                                                 #
#                                                                    #
#  Just some extra utilities to help with prep.                      #
#                                                                    #
#  Thoughts for improvements:                                        #
#                                                                    #
######################################################################


import json
import sys

def checkObservable(keys, ntuple_name, ML_name):
    """
    Returns ntuple name if it's in the list of keys, and if not it returns ML name if it's in the list of keys. 
    If neither are in the list of keys, it writes a message and exits.
    
        Parameters:
            keys (list of str): List of observable keys.
            ntuple_name (str): Name of the observable in the original root file (usually grabbed from config file).
            ML_name (str): Name of the observable as it will be used for machine learning purposes.
    """
    
    if (ntuple_name in keys):
        return ntuple_name
    elif (ML_name in keys):
        return ML_name
    else:
        print('Observable '+ML_name+' was not in root file. Exiting program.')
        sys.exit(0)
        
    
def getBranchNames(cfile):
    """
    Helps to read out the branch names from the variable names config file.
    
        Parameters:
            cfile (str): Name (including path) of the variable names config file.
            
        Returns:
            nominal name (str), systematic up name (str), systematic down name (str)
    """
    
    # Load the file
    var_names = json.load(open(cfile))
    
    # Get the three banch names
    nom = var_names["branches"]["nominal"]
    up = var_names["branches"]["sysUP"]
    down = var_names["branches"]["sysDOWN"]
    
    return nom, up, down
    
    
    
def getObservableName(cfile,keys,ML_name):
    """
    Helps to read out the ntuple name of a variable from the variable names config file.
    
        Parameters:
            cfile (str): Name (including path) of the variable names config file.
            keys (list of str): List of observable keys.
            ML_name (str): Name of the variable used by TRecNet.
            
        Returns:
            variable name (str)
    """
    
    # Load the file and observable name
    var_names = json.load(open(cfile))
    ntuple_name = var_names["observables"][ML_name]
    correct_name = checkObservable(keys, ntuple_name, ML_name)
    
    return correct_name


def getObservableNames(cfile, keys, *ML_names):
    """
    Helps to read out multiple ntuple variables names from the variable names config file.
    
        Parameters:
            cfile (str): Name (including path) of the variable names config file.
            keys (list of str): List of observable keys.
            ML_names (str): Names of the variables used by TRecNet.
            
        Returns:
            variable name (str)
    """
    
    names = [getObservableName(cfile, keys, ML_name) for ML_name in ML_names]
    
    return tuple(name for name in names)

def getObservableNamesDict(cfile, keys, *ML_names):
    """
    Helps to read out multiple ntuple variables names from the variable names config file. Outputs in list format.
    
        Parameters:
            cfile (str): Name (including path) of the variable names config file.
            keys (list of str): List of observable keys.
            ML_names (str): Names of the variables used by TRecNet.
            
        Returns:
            variable name (dict of str)
    """
    
    names = {ML_name : getObservableName(cfile, keys, ML_name) for ML_name in ML_names}
    
    return names