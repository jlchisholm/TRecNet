######################################################################
#                                                                    #
#  FileSplitter.py                                                   #
#  Author: Jenna Chisholm                                            #
#  Updated: Aug.29/25                                                #
#                                                                    #
#  A class for splitting root files into smaller root files.         #
#                                                                    #
#  Thoughts for improvements:                                        #
#                                                                    #
######################################################################


import uproot
import vector
import os, sys
from argparse import ArgumentParser
from Util import *
from JetMatcher import *


def split_file(input_file, save_dir, var_conf, max_events):
    """
    Splits a file into <num_new> new files.
    
        Parameters:
            input_file (str): path and name of the input file.
            save_dir (str): directory where the files will be saved.
            var_conf (str): path and name of the json variable config file.
            max_events (int): maximum number of events in the new files.

    """
    
    # File names
    in_name = os.path.split(input_file)[1]
    out_name = save_dir+'/'+in_name.split('.root')[0]
    
    # Get the branch names
    nom_name, up_name, down_name = getBranchNames(var_conf)

    # Open the original file
    print('Opening root file ...')
    og_file = uproot.open(input_file)
    
    # Save the keys
    #down_keys = og_file[down_name].keys()
    #up_keys = og_file[up_name].keys()
    nom_keys = [key for key in og_file[nom_name].keys() if 'TLV' not in key] # currently TLV branches with sub-branches that are causing issues
    
    # Get the reco and parton trees from the original file
    #down_tree = og_file[down_name].arrays()
    #up_tree = og_file[up_name].arrays()
    nom_tree = og_file[nom_name].arrays(nom_keys)
    
    # Close the original file
    og_file.close()
    
    # Divide
    num_events = len(nom_tree)
    curr_num = 0
    i = 0
    while(curr_num <= num_events):
        print('Creating file '+str(i)+'...')
        new_file = uproot.recreate(out_name+'_'+str(i)+'.root')
        cut_tree = nom_tree[curr_num:curr_num+max_events,]
        new_file[nom_name] = {key:cut_tree[key] for key in nom_keys}
        new_file.close()
        print('Saved file: '+out_name)
        curr_num+=max_events+1
        i+=1
    


# ---------- GET ARGUMENTS FROM COMMAND LINE ---------- #      
        
# Create the main parser
parser = ArgumentParser()

# Define arguments for EventRemover
parser.add_argument('--input',help='Input file (including path).',required=True)
parser.add_argument('--save_dir',help='Path for directory where file will be saved.',required=True)
parser.add_argument('--var_conf',help='Config file (including path) for names of variables.',required=True)
parser.add_argument('--max_events',help='Max events to include in a file.',default=10000,type=int)


# Parse the arguments and proceed with stuff
args = parser.parse_args()
split_file(args.input,args.save_dir,args.var_conf,args.max_events)


print('Done :)')
        