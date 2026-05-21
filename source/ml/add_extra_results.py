##########################################################################
#                                                                        #
#  add_extra_results.py                                                  #
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
import json
import numpy as np
import uproot
import vector
from argparse import ArgumentParser


def add_new_variable(par, obs, new_tree):
    var = par+"_"+obs
    
    # Anything that will need the 4-vector to be created
    if obs in ["px", "py"]:
        var_vecs = vector.array({"pt":new_tree[par+"_pt"], "eta": new_tree[par+"_eta"],"phi": new_tree[par+"_phi"],"m":new_tree[par+"_m"]})
        if obs=="px":
            new_tree[var] = var_vecs.px
        elif obs=="py":
            new_tree[var] = var_vecs.py
        elif obs=="e":
            new_tree[var] = var_vecs.e
        elif obs=="y":
            new_tree[var] = var_vecs.rapidity
            
    # Things that don't need the 4-vector
    else:
        if var=="ttbar_HT":
            new_tree[var] = new_tree["th_pt"]+new_tree["tl_pt"]
        elif var=="ttbar_dphi":
            new_tree[var] = np.abs((new_tree["th_phi"]-new_tree["tl_pt"])% (2*np.pi))
        
        
        
    
    return new_tree





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
                if (par+"_"+obs in new_tree.keys()):
                    print(par+"_"+obs+' already in this tree. Skipping.')
                else:
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
    
    
    