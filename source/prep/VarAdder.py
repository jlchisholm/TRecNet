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
import vector
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
        
    def add_isSemiLep_var(self, tree, keys):
        """
        Adds a binary tag to each event for whether or not the event is semi-leptonic.
        """
        
        # add isSemiLep key
        semilep_key = "isSemiLep"
        keys.append(semilep_key)

        # determine which events are semi-leptonic
        str_t_is_lep = getObservableName(self.var_conf, "t_is_lep")
        str_tbar_is_lep = getObservableName(self.var_conf, "tbar_is_lep")
        semilep_tag = tree[str_t_is_lep]!=tree[str_tbar_is_lep]

        # set semi lep tag in tree
        tree[semilep_key] = semilep_tag
        
        return tree, keys
        
        
    def add_jet_n_var(self, tree, keys):
        """
        Takes an input root tree and corresponding keys and adds jet_n variable.
        
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
        str_jet_pt = getObservableName(self.var_conf, "jet_pt")
        n_jets = [len(jet_pt) for jet_pt in tree[str_jet_pt]]

        # set n_jets in tree
        tree[jet_key] = n_jets
        
        return tree, keys
      
      
    def add_jet_isbtag(self, tree, keys, WP_id):
        """
        Gives jets a binary tag to say whether or not they are b-jets (based on the working point, <WP_id>).
        """
        
        # add jet_isbtag key
        btag_key = "jet_isbtag"
        keys.append(btag_key)
        
        # construct array of jet btags, using the jet btag index
        str_jet_btag_index = getObservableName(self.var_conf, "jet_btag_index")
        btags = tree[str_jet_btag_index].array() >= WP_id
        
        # set btags in tree
        tree[btag_key] = btags
        
        return tree, keys

    def add_hadlep_vars(self, tree, keys):
        """
        Takes an input root tree and corresponding keys, and adds th and tl (etc.) distinguishments for truth values depending on t and tbar.

            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal tree

            Returns:
                tree (root tree): nominal tree with updated truth values
                keys: fixed keys for nominal tree
        """
        
        # Specify range of events
        range_events = range(len(tree["eventNumber"]))
        
        # Get the keys for ttbar
        ttbar_keys = {}
        for p in ['t_','tbar_','w_t_','w_tbar_','w_t_decay1_','w_t_decay2_','w_tbar_decay1_','w_tbar_decay_2_','b_t_','b_tbar_']:
            for v in ['pt','eta','phi','m']:
                ttbar_keys[p+v] = getObservableName(self.var_conf,p+v)
                
        # Select events where t_is_leptonic
        str_t_is_lep = getObservableName(self.var_conf, "t_is_lep")
        sel = np.where(tree[str_t_is_lep])
                
        # Add new vars to tree
        for v in ['pt','eta','phi','m']:
            
            # leptonic
            tree['tl_'+v] = np.where(sel, tree[ttbar_keys['t_'+v]], tree[ttbar_keys['tbar_'+v]])
            tree['wl_'+v] = np.where(sel, tree[ttbar_keys['w_t_'+v]], tree[ttbar_keys['w_tbar_'+v]])
            tree['bl_'+v] = np.where(sel, tree[ttbar_keys['b_t_'+v]], tree[ttbar_keys['b_tbar_'+v]])
            keys.extend(['tl_'+v,'wl_'+v,'bl_'+v])
            
            # hadronic
            tree['th_'+v] = np.where(sel, tree[ttbar_keys['tbar_'+v]],tree[ttbar_keys['t_'+v]])
            tree['wh_'+v] = np.where(sel, tree[ttbar_keys['w_tbar_'+v]],tree[ttbar_keys['w_t_'+v]])
            tree['wh_decay1_'+v] = np.where(sel, tree[ttbar_keys['w_tbar_decay1_'+v]],tree[ttbar_keys['w_t_decay1_'+v]]) 
            tree['wh_decay2_'+v] = np.where(sel, tree[ttbar_keys['w_tbar_decay2_'+v]],tree[ttbar_keys['w_t_decay2_'+v]]) 
            tree['bh_'+v] = np.where(sel, tree[ttbar_keys['b_tbar_'+v]], tree[ttbar_keys['b_t_'+v]])
            keys.extend(['th_'+v,'wh_'+v,'wh_decay1_'+v,'wl_decay2_'+v,'bh_'+v])
            
        # Also want the pdgIds for the decays
        tree['wh_decay1_pdgid'] = np.where(sel, tree[ttbar_keys['w_tbar_decay1_pdgid']],tree[ttbar_keys['w_t_decay1_pdgid']]) 
        tree['wh_decay2_pdgid'] = np.where(sel, tree[ttbar_keys['w_tbar_decay2_pdgid']],tree[ttbar_keys['w_t_decay2_pdgid']]) 
        keys.extend(['wh_decay1_pdgid','wh_decay2_pdgid'])
    
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
            tree['b1_'+v] = np.where(sel, tree[bbbar_keys['b_'+v]], tree[bbbar_keys['bbar_'+v]])
            keys.append('b1_'+v)
            
            # Other b (b2)
            tree['b2_'+v] = np.where(sel, tree[bbbar_keys['bbar_'+v]],tree[bbbar_keys['b_'+v]])
            keys.append('b2_'+v)
    
        return tree, keys
        
    
    def addVars(self, add_isSemiLep, add_jet_n, add_jet_isbtag, add_jet_isTruth, add_hadlep, add_b1b2):
        """
        Adds desired variables to the root file.
        
            Parameters:
                add_isSemiLep (bool): Flag for adding a tag for whether or not the event is semi-leptonic.
                add_jet_n (bool): Flag for adding jet_n (number of jets in event) variable.
                add_jet_isbtag (bool): Flag for adding jet_isbtag variable (with 2 as the working default).
                add_jet_isTruth (bool): Flag for adding jet truth tags.
                add_hadlep (bool): Flag for adding all ttbar variables in terms of th and tl.
                add_b1b2 (bool): Flag for adding all bbbar variables in terms of b1 and b2.
        """
    
        # Separate input file name and its path
        #in_path = os.path.split(self.input_file)[0]
        in_name = os.path.split(self.input_file)[1]

        # Open the original file and a new file to write to
        print('Opening root file ...')
        og_file = uproot.open(self.input_file)
        
        # Get the branch names
        nom_name, up_name, down_name = getBranchNames(self.var_conf)

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
        
        
        # Add stuff
        if (add_isSemiLep):
            print('Adding isSemiLep...')
            nom_tree, nom_keys = self.add_isSemiLep(nom_tree,nom_keys)
        
        if (add_jet_n):
            print('Adding jet_n...')
            nom_tree, nom_keys = self.add_jet_n_var(nom_tree,nom_keys)
            
        if (add_jet_isbtag):
            print('Adding jet_isbtag...')
            WP_id = 2 # hard coded for now...
            nom_tree, nom_keys = self.add_jet_isbtag(nom_tree,nom_keys)
            
        if (add_hadlep):
            print('Adding thtl ...')
            nom_tree, nom_keys = self.add_thtl_vars(nom_tree,nom_keys)
            
        if (add_b1b2):
            print('Adding b1b2 ...')
            nom_tree, nom_keys = self.add_b1b2_vars(nom_tree,nom_keys)
            
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


        # Write the trees to the file
        print('Writing trees to new file...')
        new_file_name = self.save_dir+'/'+in_name
        new_file = uproot.recreate(new_file_name)
        #new_file[down_name] = {key:down_tree[key] for key in down_keys}
        #new_file[up_name] = {key:up_tree[key] for key in up_keys}
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
parser.add_argument('--add_isSemiLep',help='Add binary tag for whether or not event is semi-leptonic.',action='store_false')
parser.add_argument('--add_njets',help='Add variable for number of jets.',action='store_false')
parser.add_argument('--add_jet_isbtag',help='Add binary btags to jets.',action='store_false')
parser.add_argument('--add_jet_isTruth',help='Add binary truth tags to jets.',action='store_false')
parser.add_argument('--add_hadlep',help="Add tops, b's from tops, W's from tops, and W decays in terms of had and lep.",action='store_false')
parser.add_argument('--add_b1b2',help='Add b1 (leading) and b2 variables.',action='store_false')


# Parse the arguments and proceed with stuff
args = parser.parse_args()
adder = VarAdder(args.input,args.save_dir,args.var_conf)
adder.addVars(args.add_isSemiLep,args.add_njets,args.add_jet_isbtag,args.add_jet_isTruth,args.add_hadlep,args.add_b1b2)


print('Done :)')