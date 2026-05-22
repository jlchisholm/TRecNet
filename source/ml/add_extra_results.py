##########################################################################
#                                                                        #
#  add_extra_results.py                                                  #
#  Author: Jenna Chisholm                                                #
#  Updated: May.22/26.                                                   #
#                                                                        #
#  Adds extra variables to root result files produced by                 #
#  run_prediction.py.                                                    # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

import os
import json
import numpy as np
import uproot
import vector
from argparse import ArgumentParser

# Numpy dot product has huge memory issues, so we'll just make our own
def dot(v1, v2):
    """
    Computes the dot product of two vectors (or arrays of vectors).

        Parameters:
            v1 (np.array): Array of vectors.
            v2 (np.array): Array of vectors.
        
        Returns:
            _ (np.array): Array of dot products of v1 and v2 vectors.
    """
    
    return np.sum(v1*v2,axis=1)


def add_new_variable(par, obs, tree):
    """
    Adds new variable to the given tree.

        Parameters:
            par (str): Particle name (e.g. 'th').
            obs (str): Observable name (e.g. 'px').
            tree (np.array): Tree to add variable to.
            
        Returns:
            tree (np.array): Tree with the new variable added to it.
    """
    
    
    # Check to make sure variable isn't already in the tree
    var = par+"_"+obs
    if (var in tree.keys()):
        print(var+' already in this tree. Skipping.')
        return tree
    
    # Anything that will need the 4-vector to be created
    elif obs in ["px", "py", "E", "y", "pout"]:
        var_vecs = vector.array({"pt":tree[par+"_pt"], "eta": tree[par+"_eta"],"phi": tree[par+"_phi"],"m":tree[par+"_m"]})
        if obs=="px":
            tree[var] = var_vecs.px
        elif obs=="py":
            tree[var] = var_vecs.py
        elif obs=="E":
            tree[var] = var_vecs.e
        elif obs=="y":
            tree[var] = var_vecs.rapidity
        elif var=="th_pout":
            tl_vecs = vector.array({"pt":tree["tl_pt"], "eta": tree["tl_eta"],"phi": tree["tl_phi"],"m":tree["tl_m"]})
            th_px, th_py, th_pz = var_vecs.px, var_vecs.py, var_vecs.pz
            tl_px, tl_py, tl_pz = tl_vecs.px, tl_vecs.py, tl_vecs.pz
            th_P = np.stack([th_px, th_py, th_pz], axis=1)
            tl_P = np.stack([tl_px, tl_py, tl_pz], axis=1)
            ez = np.repeat(np.array([[0,0,1]]), tree["th_pt"].shape[0],axis=0)
            tree[var] = dot(th_P, np.cross(tl_P,ez))/np.linalg.norm(np.cross(tl_P,ez),axis=1)
        elif var=="tl_pout":
            th_vecs = vector.array({"pt":tree["th_pt"], "eta": tree["th_eta"],"phi": tree["th_phi"],"m":tree["th_m"]})
            tl_px, tl_py, tl_pz = var_vecs.px, var_vecs.py, var_vecs.pz
            th_px, th_py, th_pz = th_vecs.px, th_vecs.py, th_vecs.pz
            tl_P = np.stack([tl_px, tl_py, tl_pz], axis=1)
            th_P = np.stack([th_px, th_py, th_pz], axis=1)
            ez = np.repeat(np.array([[0,0,1]]), tree["th_pt"].shape[0],axis=0)
            tree[var] = dot(tl_P, np.cross(th_P,ez))/np.linalg.norm(np.cross(th_P,ez),axis=1)
            
    # Things that don't need the 4-vector
    else:
        if var=="ttbar_Ht":
            tree[var] = tree["th_pt"]+tree["tl_pt"]
        elif var=="ttbar_dphi":
            tree[var] = np.abs((tree["th_phi"]-tree["tl_pt"])% (2*np.pi))
        elif var=="ttbar_deta":
            tree[var] = np.abs((tree["th_eta"]-tree["tl_eta"]))
        elif var=="ttbar_ystar":
            if "th_y" not in tree.keys():
                add_new_variable("th","y",tree)
            if "tl_y" not in tree.keys():
                add_new_variable("tl","y",tree)
            tree[var] = np.abs((tree["th_y"]-tree["tl_y"])/2)    
        elif var=="ttbar_yboost":
            if "th_y" not in tree.keys():
                add_new_variable("th","y",tree)
            if "tl_y" not in tree.keys():
                add_new_variable("tl","y",tree)
            tree[var] = np.abs((tree["th_y"]+tree["tl_y"])/2)
        elif var=="ttbar_chi":
            if "ttbar_ystar" not in tree.keys():
                add_new_variable("ttbar","ystar",tree)
            tree[var] = np.exp(2*np.abs(tree["ttbar_ystar"]))
        
    
    return tree





### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Get JSON file name of extra var info and results file from command line and load the file
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_results', help='Root file with results to which you want to add new variables.', type=str,required=True)
    parser.add_argument('-c','--extra_var_config', help='JSON file name (including path) that contains the extra variables you want to add.', type=str, required=True)
    parser.add_argument('-t','--tree_names', help='Tree names to add the variables to, each separated by a space and enclosed by quotation marks (i.e. "reco parton").', type=str, required=True)
    args = parser.parse_args()
    f_results = args.input_results
    f_extra_var_config = args.extra_var_config
    tree_names = args.tree_names.split()
    result_path = os.path.split(f_results)[0]
    
    # Read in the extra variables desired
    with open(f_extra_var_config) as f:
        extra_vars = json.load(f)
        
    # Open the original file and a new file to write to
    print('Opening root file ...')
    og_file = uproot.open(f_results)
    og_tree_names = og_file.keys()
    
    # Create new file
    print('Creating new root file ...')
    new_file_name = result_path+'/'+'Results+Extra.root'
    new_file = uproot.recreate(new_file_name)
    
    # Add new variables to the desired trees
    for tree_name in tree_names:
        print('Adding new variables to '+tree_name+' tree ...')
        
        # Copy the original tree to the new file
        new_tree = og_file[tree_name].arrays(library="np") #.arrays(og_file[tree].keys())
        
        # Add the variables
        for par, obs_list in extra_vars.items():
            for obs in obs_list:
                add_new_variable(par,obs,new_tree)
                
        # Write the new tree to the file
        print('Writing '+tree_name+' tree to new file...')
        new_file.mktree(tree_name, new_tree)
        
    # For trees not being updated, we still want them in the new file
    for og_tree_name in og_tree_names:
        og_tree_name = og_tree_name[:-2] # remove the "1;" at the end of tree names
        if og_tree_name not in tree_names:
            print('Writing '+og_tree_name+' tree to new file...')
            new_file.mktree(og_tree_name, og_file[og_tree_name].arrays(library="np"))

    # Finish up
    print('Saved file: '+new_file_name)
    new_file.close()