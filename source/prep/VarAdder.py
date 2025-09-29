######################################################################
#                                                                    #
#  VarAdder.py                                                       #
#  Author: Jenna Chisholm                                            #
#  Updated: Aug.29/25                                                #
#                                                                    #
#  A class for adding desirabled observables, for training           #
#  purposes.                                                         #
#                                                                    #
#  Thoughts for improvements:                                        #
#       - Separate variables (tags) for when a jet is from tt,       #
#         tt+jets, ttbb, or is something else.                       #
#                                                                    #
######################################################################

import uproot
import os, sys
from argparse import ArgumentParser
import json
from Util import *
from JetMatcher import *


class VarAdder:
    """ 
    A class for adding desired observables to a root file.

        Methods:
            add_isSemiLep_var: adds a binary tag to each event for whether or not the event is semi-leptonic.
            add_jet_n_var: adds variable for the number of jets in each event.
            add_jet_isbtag_var: adds jets a binary tag to say whether or not they are b-jets (based on the working point, <WP_id>).
            add_bjet_n_var: adds variable for the number of bjets in each event.
            add_hadlep_vars: adds th and tl (etc.) distinguishments for truth values depending on t and tbar (etc.).
            add_b1b2_vars: adds b1 and b2 distinguishments for truth values depending on b and bbar.
            addVars: adds desired variables to given root file.

    """

    def __init__(self,input_file,save_dir,var_conf,var_adder_conf):
        print("Creating VarAdder.")
        self.input_file = input_file
        self.save_dir = save_dir
        self.var_conf = var_conf
        self.read_in_settings(var_adder_conf)
 
    def read_in_settings(self,var_adder_conf):
        
        # Read in the var adder / settings config file
        with open(var_adder_conf) as f:
            settings = json.load(f)
        
        # Read in which variables we'll be adding
        self.add_isSemiLep = settings["isSemiLep"]["add_var"]
        self.add_jet_n = settings["jet_n"]["add_var"]
        self.add_jet_isbtag = settings["jet_isbtag"]["add_var"]
        self.add_bjet_n = settings["bjet_n"]["add_var"]
        self.add_hadlep = settings["hadlep"]["add_var"]
        self.add_b1b2 = settings["b1b2"]["add_var"]
        self.add_jet_isTruth = settings["jet_isTruth"]["add_var"]
        self.add_jet_isExtraB = settings["jet_isExtraB"]["add_var"]
        self.add_jet_isFromttbar = settings["jet_isFromttbar"]["add_var"]
        
        # Read in additional settings as necessary
        if(self.add_jet_isbtag):
            self.WP_id = settings["jet_isbtag"]["WP_id"]
        # if(self.add_jet_isTruth):
        #     self.dR_cut = settings["jet_isTruth"]["dR_cut"]
        #     self.allow_double_matching = settings["jet_isTruth"]["allow_double_matching"]
        #     self.b_mode = settings["jet_isTruth"]["b_mode"]
        
        
        
    def add_isSemiLep_var(self, tree, keys):
        """
        Adds a binary tag to each event for whether or not the event is semi-leptonic.
        
            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree
        """
        
        # add isSemiLep key
        semilep_key = "isSemiLep"
        keys.append(semilep_key)

        # determine which events are semi-leptonic
        str_t_is_lep = getObservableName(self.var_conf, keys, "t_is_lep")
        str_tbar_is_lep = getObservableName(self.var_conf, keys, "tbar_is_lep")
        semilep_tag = tree[str_t_is_lep]!=tree[str_tbar_is_lep]

        # set semi lep tag in tree
        tree[semilep_key] = semilep_tag
        
        return tree, keys
        
        
    def add_jet_n_var(self, tree, keys):
        """
        Adds variable for the number of jets in each event.
        
            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree
        """
        
        # add n_jets key
        jet_key = "jet_n"
        keys.append(jet_key)

        # construct array of jet numbers based on jet_pt
        str_jet_pt = getObservableName(self.var_conf, keys, "jet_pt")
        n_jets = [len(jet_pt) for jet_pt in tree[str_jet_pt]]

        # set n_jets in tree
        tree[jet_key] = n_jets
        
        return tree, keys
      
      
    def add_jet_isbtag_var(self, tree, keys):
        """
        Adds jets a binary tag to say whether or not they are b-jets (based on the working point, <WP_id>).
        
            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree
        """
        
        # add jet_isbtag key
        btag_key = "jet_isbtag"
        keys.append(btag_key)
        
        # construct array of jet btags, using the jet btag index
        str_jet_btag_index = getObservableName(self.var_conf, keys, "jet_btag_index")
        btags = tree[str_jet_btag_index] >= self.WP_id
        
        # set btags in tree
        tree[btag_key] = btags
        
        return tree, keys
    
    def add_bjet_n_var(self, tree, keys):
        """
        Adds variable for the number of bjets in each event.
        
            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree
        """
        
        # add n_jets key
        bjet_key = "bjet_n"
        keys.append(bjet_key)

        # construct array of jet numbers based on jet_pt
        str_jet_isbtag = getObservableName(self.var_conf, keys, "jet_isbtag")
        btags = tree[str_jet_isbtag]
        n_bjets = [len(bjets) for bjets in btags[btags==True]]

        # set n_jets in tree
        tree[bjet_key] = n_bjets
        
        return tree, keys 
 
    def add_hadlep_vars(self, tree, keys):
        """
        Adds th and tl (etc.) distinguishments for truth values depending on t and tbar (etc.).

            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree
        """
        
        # Get the keys for ttbar
        ttbar_keys = {}
        for p in ['t_','tbar_','w_t_','w_tbar_','w_t_decay1_','w_t_decay2_','w_tbar_decay1_','w_tbar_decay2_','b_t_','bbar_tbar_']:
            for v in ['pt','eta','phi','m']:
                ttbar_keys[p+v] = getObservableName(self.var_conf, keys, p+v)
                
        # Select events where t_is_leptonic
        str_t_is_lep = getObservableName(self.var_conf, keys, "t_is_lep")
                
        # Add new vars to tree
        for v in ['pt','eta','phi','m']:
            
            # leptonic
            tree['tl_'+v] = np.where(tree[str_t_is_lep], tree[ttbar_keys['t_'+v]], tree[ttbar_keys['tbar_'+v]])
            tree['wl_'+v] = np.where(tree[str_t_is_lep], tree[ttbar_keys['w_t_'+v]], tree[ttbar_keys['w_tbar_'+v]])
            tree['bl_'+v] = np.where(tree[str_t_is_lep], tree[ttbar_keys['b_t_'+v]], tree[ttbar_keys['bbar_tbar_'+v]])
            keys.extend(['tl_'+v,'wl_'+v,'bl_'+v])
            
            # hadronic
            tree['th_'+v] = np.where(tree[str_t_is_lep], tree[ttbar_keys['tbar_'+v]],tree[ttbar_keys['t_'+v]])
            tree['wh_'+v] = np.where(tree[str_t_is_lep], tree[ttbar_keys['w_tbar_'+v]],tree[ttbar_keys['w_t_'+v]])
            tree['wh_decay1_'+v] = np.where(tree[str_t_is_lep], tree[ttbar_keys['w_tbar_decay1_'+v]],tree[ttbar_keys['w_t_decay1_'+v]]) 
            tree['wh_decay2_'+v] = np.where(tree[str_t_is_lep], tree[ttbar_keys['w_tbar_decay2_'+v]],tree[ttbar_keys['w_t_decay2_'+v]]) 
            tree['bh_'+v] = np.where(tree[str_t_is_lep], tree[ttbar_keys['bbar_tbar_'+v]], tree[ttbar_keys['b_t_'+v]])
            keys.extend(['th_'+v,'wh_'+v,'wh_decay1_'+v,'wh_decay2_'+v,'bh_'+v])
            
        # Also want the pdgIds for the decays
        str_w_tbar_decay1_pdgid, str_w_t_decay1_pdgid, str_w_tbar_decay2_pdgid, str_w_t_decay2_pdgid = getObservableNames(self.var_conf, keys, 'w_tbar_decay1_pdgid','w_t_decay1_pdgid','w_tbar_decay2_pdgid','w_t_decay2_pdgid')
        tree['wh_decay1_pdgid'] = np.where(tree[str_t_is_lep], tree[str_w_tbar_decay1_pdgid],tree[str_w_t_decay1_pdgid]) 
        tree['wh_decay2_pdgid'] = np.where(tree[str_t_is_lep], tree[str_w_tbar_decay2_pdgid],tree[str_w_t_decay2_pdgid]) 
        keys.extend(['wh_decay1_pdgid','wh_decay2_pdgid'])
    
        return tree, keys
        

        
    def add_b1b2_vars(self, tree, keys):
        """
        Adds b1 and b2 distinguishments for truth values depending on b from t and bbar from tbar.

            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree
        """
        
        # Get the keys for bbbar
        bbbar_keys = {}
        for p in ['b_t_','bbar_tbar_']:
            for v in ['pt','eta','phi','m']:
                bbbar_keys[p+v] = getObservableName(self.var_conf, keys, p+v)
                
        # Select events where b_pt > bbar_pt
        sel = np.greater(tree[bbbar_keys['b_t_pt']],tree[bbbar_keys['bbar_tbar_pt']])
                
        # Add b1b2 vars to trees
        for v in ['pt','eta','phi','m']:
            
            # Leading b (b1)
            tree['b1_'+v] = np.where(sel, tree[bbbar_keys['b_t_'+v]], tree[bbbar_keys['bbar_tbar_'+v]])
            keys.append('b1_'+v)
            
            # Other b (b2)
            tree['b2_'+v] = np.where(sel, tree[bbbar_keys['bbar_tbar_'+v]],tree[bbbar_keys['b_t_'+v]])
            keys.append('b2_'+v)
    
        return tree, keys
    
    # # Previously made jet matcher doesn't work with new files, just use Ryan's jet_origin
    # def add_jet_isTruth_var(self, tree, keys, in_name):
        
    #     # Do the matching
    #     matcher = JetMatcher()
    #     isttbarJet, matching_info = matcher.getMatches(tree, keys, self.var_conf, self.dR_cut, self.allow_double_matching, self.b_mode)

    #     # Save to nominal tree
    #     print('Adding jet_isTruth...')
    #     tree['jet_isTruth'] = isttbarJet
    #     keys.append('jet_isTruth')
        
    #     # Save the matching info separately, since it's a weird shape, and save in its own folder
    #     match_file_name = self.save_dir+'/matching_info/'+in_name.split('.root')[0]+'.npy'
    #     if not os.path.exists(self.save_dir+'/matching_info'):
    #         os.mkdir(self.save_dir+'/matching_info')
    #     np.save(match_file_name,matching_info)
    #     print('Saved file: '+match_file_name)  
        
    #     return tree, keys
        
    
    def add_jet_isTruth_var(self, tree, keys): 
        """
        Adds a binary tags for whether or not a jet is associated with the ttbar system. 
        
            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree        
        """
        
        # add jet isTruth key
        jetTruth_key = "jet_isTruth"
        keys.append(jetTruth_key)

        # determine which jets are true
        str_jet_origin = getObservableName(self.var_conf, keys, "jet_origin")
        isTruthJet = tree[str_jet_origin]!=0

        # set semi lep tag in tree
        tree[jetTruth_key] = isTruthJet
        
        return tree, keys
        
    def add_jet_isExtraB_var(self, tree, keys): 
        """
        Adds a binary tag for whether or not a jet is associated with the extra b-jets in ttbb.
        
            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree        
        """
        
        # add jet tag key
        jetTag_key = "jet_isExtraB"
        keys.append(jetTag_key)

        # determine which jets are true
        str_jet_origin = getObservableName(self.var_conf, keys, "jet_origin")
        isExtraB = np.abs(tree[str_jet_origin])==5

        # set semi lep tag in tree
        tree[jetTag_key] = isExtraB
        
        return tree, keys  
    
    def add_jet_isFromttbar_var(self, tree, keys): 
        """
        Adds a binary tag for whether or not a jet is from ttbar (i.e. a b-jet from top decay, or a jet from the W decay).
        
            Parameters:
                tree (awkward array): nominal tree
                keys (list of str): keys for nominal tree

            Returns:
                tree (awkward array): nominal tree with updated truth values
                keys (list of str): fixed keys for nominal tree        
        """
        
        # add jet tag key
        jetTag_key = "jet_isFromttbar"
        keys.append(jetTag_key)

        # determine which jets are true
        str_jet_origin = getObservableName(self.var_conf, keys, "jet_origin")
        sel_b_from_t = np.abs(tree[str_jet_origin])==6
        sel_W_decay = np.abs(tree[str_jet_origin])==24
        isFromttbar = sel_b_from_t + sel_W_decay

        # set semi lep tag in tree
        tree[jetTag_key] = isFromttbar
        
        return tree, keys  
      
        
    
    def addVars(self):
        """
        Adds desired variables to given root file.
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
        if (self.add_isSemiLep):
            print('Adding isSemiLep ...')
            nom_tree, nom_keys = self.add_isSemiLep_var(nom_tree,nom_keys)
        
        if (self.add_jet_n):
            print('Adding jet_n ...')
            nom_tree, nom_keys = self.add_jet_n_var(nom_tree,nom_keys)
            
        if (self.add_jet_isbtag):
            print('Adding jet_isbtag ...')
            nom_tree, nom_keys = self.add_jet_isbtag_var(nom_tree,nom_keys)
            
        if (self.add_bjet_n):
            print('Adding bjet_n ...')
            nom_tree, nom_keys = self.add_bjet_n_var(nom_tree,nom_keys)
            
        if (self.add_hadlep):
            print('Adding thtl ...')
            nom_tree, nom_keys = self.add_hadlep_vars(nom_tree,nom_keys)
            
        if (self.add_b1b2):
            print('Adding b1b2 ...')
            nom_tree, nom_keys = self.add_b1b2_vars(nom_tree,nom_keys)
            
        if (self.add_jet_isTruth):
            print('Adding jet_isTruth ...')
            nom_tree, nom_keys = self.add_jet_isTruth_var(nom_tree, nom_keys)
            
        if (self.add_jet_isExtraB):
            print('Adding jet_isExtraB ...')
            nom_tree, nom_keys = self.add_jet_isExtraB_var(nom_tree, nom_keys)
            
        if (self.add_jet_isFromttbar):
            print('Adding jet_isFromttbar ...')
            nom_tree, nom_keys = self.add_jet_isFromttbar_var(nom_tree, nom_keys)
                   
                   
        # Write the trees to the file
        print('Writing trees to new file...')
        if self.add_jet_isbtag:
            new_file_name = self.save_dir+'/'+in_name.split('.root')[0]+'_WP'+str(self.WP_id)+'.root'
        else:
            new_file_name = self.save_dir+'/'+in_name.split('.root')[0]+'_WP'+str(self.WP_id)+'.root'
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
parser.add_argument('--var_adder_conf',help='Config file (including path) specifically for VarAdder.',required=True)


# Parse the arguments and proceed with stuff
args = parser.parse_args()
adder = VarAdder(args.input,args.save_dir,args.var_conf,args.var_adder_conf)
adder.addVars()


print('Done :)')