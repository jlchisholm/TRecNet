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
    
    
    
def getObservableName(cfile,ML_name):
    """
    Helps to read out the ntuple name of a variable from the variable names config file.
    
        Parameters:
            cfile (str): Name (including path) of the variable names config file.
            ML_name (str): Name of the variable used by TRecNet.
            
        Returns:
            variable name (str)
    """
    
    # Load the file
    var_names = json.load(open(cfile))
    name = var_names["observables"][ML_name]
    
    return name


def getObservableNames(cfile,*ML_names):
    """
    Helps to read out multiple ntuple variables names from the variable names config file.
    
        Parameters:
            cfile (str): Name (including path) of the variable names config file.
            ML_names (str): Names of the variables used by TRecNet.
            
        Returns:
            variable name (str)
    """
    
    names = [getObservableName(cfile,ML_name) for ML_name in ML_names]
    
    return tuple(name for name in names)

def getObservableNamesDict(cfile,*ML_names):
    """
    Helps to read out multiple ntuple variables names from the variable names config file. Outputs in list format.
    
        Parameters:
            cfile (str): Name (including path) of the variable names config file.
            ML_names (str): Names of the variables used by TRecNet.
            
        Returns:
            variable name (dict of str)
    """
    
    names = {ML_name : getObservableName(cfile,ML_name) for ML_name in ML_names}
    
    return names