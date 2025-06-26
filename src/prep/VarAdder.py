######################################################################
#                                                                    #
#  VarAdder.py                                                       #
#  Author: Jenna Chisholm                                            #
#  Updated: Jun.26/25                                                #
#                                                                    #
#  A class for adding desirabled observables, for training           #
#  purposes.                                                         #
#                                                                    #
#  Thoughts for improvements:                                        #
#       - Add functionality to add btags based on WP                 #
#                                                                    #
######################################################################

import uproot
import os, sys
from argparse import ArgumentParser
from Util import *
from JetMatcher import *


class VarAdder:
    """ 
    A class for adding desired observables to a root file.

        Methods:
            add_njets_var: adds variable for number of jets in each event.
            add_b1b2_vars: adds b1 and b2 distinguishments for truth values depending on b and bbar
            addVars: adds desired variables to root file.

    """

    def __init__(self,input_file,save_dir,var_conf):
        print("Creating VarAdder.")
        self.input_file = input_file
        self.save_dir = save_dir
        self.var_conf = var_conf
        
        
    def add_njets_var(self, tree, keys):
        """
        Takes an input root tree and corresponding keys and adds njets variable.
        
            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal tree

            Returns:
                tree (root tree): nominal tree with updated truth values
                keys: fixed keys for nominal tree
        
        """
        
        # add n_jets key
        jet_key = "jet_n"
        keys.append(jet_key)

        # construct array of jet numbers based on jet_pt
        n_jets = [len(jet_pt) for jet_pt in tree["jet_pt"]]

        # set n_jets in tree
        tree[jet_key] = n_jets
        
        return tree, keys
        
        

        
    def add_b1b2_vars(self, tree, keys):
        """
        Takes an input root tree and corresponding keys, and adds b1 and b2 distinguishments for truth values depending on b and bbar.

            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal tree

            Returns:
                tree (root tree): nominal tree with updated truth values
                keys: fixed keys for nominal tree
        """
        
        # Specify range of events
        range_events = range(len(tree["eventNumber"]))
        
        # Get the keys for bbbar
        bbbar_keys = {}
        for p in ['b_','bbar_']:
            for v in ['pt','eta','phi','m','y','E']:
                bbbar_keys[p+v] = getObservableName(self.var_conf,p+v)
                
        # Select events where b_pt > bbar_pt
        sel = np.greater(tree[bbbar_keys['b_pt']],tree[bbbar_keys['bbar_pt']])
                
        # Add b1b2 vars to trees
        for v in ['pt','eta','phi','m', 'y', 'E']:
            
            # Leading b (b1)
            tree['b1_'+v] = np.where(sel, tree[bbbar_keys['b_pt']], tree[bbbar_keys['bbar_pt']])
            keys.append('b1_'+v)
            
            # Other b (b2)
            tree['b2_'+v] = np.where(sel, tree[bbbar_keys['bbar_pt']],tree[bbbar_keys['b_pt']])
            keys.append('b2_'+v)
    
        return tree, keys
        
    
    def addVars(self,add_njets,add_jet_isTruth,add_b1b2):
        """
        Adds desired variables to the root file.
        
            Parameters:
                add_njets (bool): Flag for adding njets variable.
                add_jet_isTruth (bool): Flag for adding jet truth tags.
                add_b1b2 (bool): Flag for adding all bbbar variables in terms of b1 and b2.
        """
        
            
            
        # Separate input file name and its path
        #in_path = os.path.split(self.input_file)[0]
        in_name = os.path.split(self.input_file)[1]

        # Open the original file and a new file to write to
        print('Opening root file ...')
        og_file = uproot.open(self.input_file)

        # Get the reco and parton trees from the original file
        nom_name, up_name, down_name = getBranchNames(self.var_conf)
        down_tree = og_file[down_name].arrays()
        up_tree = og_file[up_name].arrays()
        nom_tree = og_file[nom_name].arrays()

        # Save the keys for later
        down_keys = og_file[down_name].keys()
        up_keys = og_file[up_name].keys()
        nom_keys = og_file[nom_name].keys()

        # Close the original file
        og_file.close()
        
        
        # Add stuff
        if (add_njets):
            print('Adding njets...')
            nom_tree, nom_keys = self.add_njets_var(nom_tree,nom_keys)
            
        if (add_jet_isTruth):
            
            print('Matching jets ...')
            
            dR_cut = 1.0 # Manually setting dR cut, could add more flexibility with this later
            allow_double_matching = True # Manually setting this too for now
            
            # Do the matching
            matcher = JetMatcher()
            isttbarJet, matching_info = matcher.getMatches(nom_tree, dR_cut, allow_double_matching)
            
            # Save to nominal tree
            print('Adding jet_isTruth...')
            nom_tree['jet_isTruth'] = isttbarJet
            nom_keys.append('jet_isTruth')
            
            # Save the matching info separately, since it's a weird shape, and save in its own folder
            match_file_name = self.save_dir+'/matching_info/'+in_name.split('.root')[0]+'.npy'
            if not os.path.exists(self.save_dir+'/matching_info'):
                os.mkdir(self.save_dir+'/matching_info')
            np.save(match_file_name,matching_info)
            print('Saved file: '+match_file_name)
            
            
        if (add_b1b2):
            print('Adding b1b2 ...')
            nom_tree, nom_keys = self.add_b1b2_vars(nom_tree,nom_keys)
            
            


        # Write the trees to the file
        print('Writing trees to new file...')
        new_file_name = self.save_dir+'/'+in_name
        new_file = uproot.recreate(new_file_name)
        new_file[down_name] = {key:down_tree[key] for key in down_keys}
        new_file[up_name] = {key:up_tree[key] for key in up_keys}
        new_file[nom_name] = {key:nom_tree[key] for key in nom_keys}

        print('Saved file: '+new_file_name)

        # Close new file
        new_file.close()
        
        
        
        
        
        
        
# ---------- GET ARGUMENTS FROM COMMAND LINE ---------- #      
        
# Create the main parser
parser = ArgumentParser()

# Define arguments for EventRemover
parser.add_argument('--input',help='Input file (including path).',required=True)
parser.add_argument('--save_dir',help='Path for directory where file will be saved.',required=True)
parser.add_argument('--var_conf',help='Config file (including path) for names of variables.',required=True)
parser.add_argument('--add_njets',help='Add variable for number of jets.',action='store_false')
parser.add_argument('--add_jet_isTruth',help='Add binary truth tags to jets.',action='store_false')
parser.add_argument('--add_b1b2',help='Add b1 (leading) and b2 variables.',action='store_false')


# Parse the arguments and proceed with stuff
args = parser.parse_args()
adder = VarAdder(args.input,args.save_dir,args.var_conf)
adder.addVars(args.add_njets,args.add_jet_isTruth,args.add_b1b2)


print('Done :)')