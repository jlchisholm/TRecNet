######################################################################
#                                                                    #
#  ttbbPrep.py                                                       #
#  Author: Dean Ciarniello                                           #
#  Updated: May 19th, 2023                                           #
#                                                                    #
#  For preparing ttbb root files to be converted to H5 files         #
#  to then be run in TRecNet                                         #
#                                                                    #
######################################################################

# Import Statements
import uproot
import numpy as np
import os
import vector
from argparse import ArgumentParser
from __main__ import *


class ttbbCutter:
    """
    A class for cutting out non-useful events from nominal root tree. This also adds jet number per event information (if its missing) and boolean b-tagging information (if its missing)

        Methods:
            cutTree: removes non-useful events from trees
            ttbbCut: takes input root file with nominal tree, makes cuts, and returns new root file cut nominal tree
            addNumJets: takes input root tree and adds the number of jets per event (based on jet_pt)
            addBinaryBTags: takes input root tree and adds a branch with boolean values for each jet (1 if b tagged, 0 it not)
            addLepVals: combines el_ and mu_ pt, eta, and phi branches to make lep_ pt, eta, and phi branch in tree
    """

    # Constructor
    def __init__(self):
        print("Creating ttbbCut")


    def cutTree(self, tree, num_b_tags, n_jets_min, DL1r_op_point):
        """
        Implements cuts on input tree (from root file) on the minimum number of b tags and minimum number of jets

            Parameters:
                tree (root tree): nominal tree from root file
                num_b_tags (int): minimum number of b tags
                n_jets_min (int): minumum number of jets

            Returns:
                tree (root tree): nominal tree with bad events removed
        """

        # minimin num_b_tags b tagged jets and n_jets_min total jets
        '''
        DL1r Weights:
            1: 0-60%
            2: 60-70%
            3: 70-77%
            4: 77-85%
            5: 85-100%
        '''
        b_tags = [[(weight >= DL1r_op_point) for weight in DL1r_weights] for DL1r_weights in tree["jet_tagWeightBin_DL1r_Continuous"]] # cut on DL1r weights given DL1r operating point
        b_sel = [(sum(weights)>= num_b_tags) for weights in b_tags] # add up the binary value of b-tags per event and check if >= num_b_tags
        print("cutting on b tags")
        
        j_sel = [(len(jet_pt)>= n_jets_min) for jet_pt in tree["jet_pt"]] # should also be able to cut on n_jets
        print("cutting on jet number")

        # combine selections into one (i.e. if both selections are satisfied)
        sel = np.logical_and(j_sel, b_sel)
        tree = tree[sel]

        # returns 'cut' tree
        print('cut done')
        return tree


    def addNumJets(self, tree, keys):
        """
        Adds the number of jets per event to the tree/keys

            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal root tree

            Returns:
                tree (root tree): nominal tree with n_jets added
                keys: keys for nominal tree with'n_jet' added
        """

        # add n_jets key
        keys.append("n_jets")

        # construct array of jet numbers based on jet_pt
        n_jets = [len(jet_pt) for jet_pt in tree["jet_pt"]]

        # set n_jets in tree
        tree['n_jets'] = n_jets

        return tree, keys

    def addBinaryBTags(self, tree, keys, DL1r_op_point):
        """
        Adds an array of binary values corresponding to whether or not a jet is b-tagged

            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal root tree

            Returns:
                tree (root tree): nominal tree with binary b-tags added
                keys: keys for nominal tree with binary b-tags added
        """

        # add n_jets key (for 70% 'operating point')
        keys.append("jet_isbtagged_DL1r_70") # Hardcoded @ 70% for now, to match Jenna's TRecNet Model

        # construct array of jet numbers based on jet_pt (for 70% 'operating point')
        b_tags = [[(weight >= DL1r_op_point) for weight in DL1r_weights] for DL1r_weights in tree["jet_tagWeightBin_DL1r_Continuous"]]

        # set n_jets in tree
        tree['jet_isbtagged_DL1r_70'] = b_tags

        return tree, keys
    

    def addLepVals(self, tree, keys):
        """
        Combines branches for el_ and mu_ pt, eta, and phi so that there are lep_ pt, eta, and phi branches

            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal root tree

            Returns:
                tree (root tree): nominal tree with lep_ branches added
                keys: keys for nominal tree with with lep_ branches added
        """
        # loop over pt, eta, and phi
        for var in ["pt", "eta", "phi"]:
            # add lep_var to keys
            keys.append("lep_"+var)

            # combine el_ and mu_ branches to make lep_ branch and add to tree (assumes events are semi-leptonic)
            lep_var = [mu if (len(el)==0) else el for el, mu in zip(tree["el_"+var], tree["mu_"+var])]
            tree["lep_"+var] = lep_var

            print("added: lep_" + var)

        return tree, keys

    
    def ttbbCut(self, root_file, save_dir, tree_name, num_b_tags, n_jets_min, DL1r_op_point):
        """
        Copies nominal tree from input root file, makes cuts on minimum number of jets and minimum number of b tags,
        and makes new root file with the cut tree.
        
            Parameters:
                root_file: the full path to the input root file
                save_dir: the path to the desired save directory
                tree_name: the name of the nominal tree in the input root file (i.e. nominal_Loose)
                num_b_tags (int): minimum number of b tags
                n_jets_min (int): minumum number of jets
                DL1r_op_point (int): corresponds to the operating point of the DL1r b-tagging network
        """

        in_name = os.path.split(root_file)[1]
        og_file = uproot.open(root_file)
        fix_file_path = save_dir+'/'+in_name.split('.root')[0]+'_ttbb_cut_'+str(num_b_tags)+'_'+str(n_jets_min)+'_'+str(DL1r_op_point)+'.root'
        fix_file = uproot.recreate(fix_file_path)

        # save keys and trees from original file
        nom_tree = og_file[tree_name].arrays()
        nom_keys = og_file[tree_name].keys()

        # close orginal file
        og_file.close()
        
        # add n_jets parameter to key+array if n_jets parameter not included
        if "n_jets" not in nom_keys:
            print("Adding number of jets")
            nom_tree, nom_keys = self.addNumJets(nom_tree, nom_keys)

        # add binary "is b-tagged" branch if not already included (currently hardcoded to 70% operating point to match Jenna's TRecNet setup)
        if "jet_isbtagged_DL1r_70" not in nom_keys:
            print("Adding binary 'is b-tagged' values")
            nom_tree, nom_keys = self.addBinaryBTags(nom_tree, nom_keys, DL1r_op_point)

        # add lepton information if not already included in tree
        if "lep_pt" not in nom_keys:
            print("Combining el_ and mu_ to get lep_")
            nom_tree, nom_keys = self.addLepVals(nom_tree, nom_keys)

        # cut tree events
        print('cutting tree')
        nom_tree = self.cutTree(nom_tree, num_b_tags, n_jets_min, DL1r_op_point)

        # remove problematic keys i.e. keys from the ttbbVars branch (caused field not found errors... why?)
        # TODO: figure out why this is a problem and fix, to preserve all information from original root tree (although this is not necessary to run TRecNet)
        nom_keys_fix= [key for key in nom_keys if 'ttbbVars' not in key.split('/')]

        #print(nom_keys_fix)

        # add tree back to fixed root file
        fix_file[tree_name] = {key: nom_tree[key] for key in nom_keys_fix}
        
        # save and close fixed root file
        print('Saved file: '+fix_file_path)
        fix_file.close()



"""
NOTES for tt+bb changes

ROOT File Naming Conventions (old -> tt+bb):
    jet_n       ->  n_jets
    jet_phi     ->  jet_phi
    jet_pt      ->  jet_pt
    jet_e       ->  jet_e
    jet_eta     ->  jet_eta

"""
class keyConverter:
    """
    A class for converting the keys of the ttbb root tree to match the keys in TRecNet

        Methods:
            updateKeys: replaces 'old' keys with 'new' keys in tree and keys, if 'old' key is in the current list of keys
            convertKeys: takes an input root files, updates the keys for the root tree accordint to key_changes, and saves 
                new root file with fixed keys
    """

    # Constructor
    def __init__(self):
        print("Creating keyConverter")
        self.key_changes ={"n_jets": "jet_n"} #dictionary of keys to change {"old_key1": "new_key1", ...}

    
    def updateKeys(self, tree, keys):
        """
        Replaces the old keys in self.key_changes with the new keys in self.key_changes, if the old key is in the input list of keys

            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal tree

            Returns:
                tree (root tree): nominal tree with fixed keys
                keys: fixed keys for nominal tree
        """

        for old, new in self.key_changes.items():
            # skip if old key not in list of keys
            if old not in keys: continue

            # Add new key, update array, and remove old key
            keys.append(new)
            tree[new] = tree[old]
            keys.remove(old) 

        return tree, keys


    def convertKeys(self, root_file, save_dir, tree_name):
        """
        Takes an input root file, updates the keys in tree according to self.key_changes, and saves a new root file with the 
        updated tree

            Parameters:
                root_file: the full path to the input root file
                save_dir: the path to the desired save directory
                tree_name: the name of the nominal tree in the input root file (i.e. nominal_Loose)
        """

        in_name = os.path.split(root_file)[1]
        og_file = uproot.open(root_file)
        fix_file = uproot.recreate(save_dir+'/'+in_name.split('.root')[0]+'_fixed_keys'+'.root')

        # save keys and trees from original file
        nom_tree = og_file[tree_name].arrays()
        nom_keys = og_file[tree_name].keys()

        # close orginal file
        og_file.close()

        # update keys
        print('updating keys')
        nom_tree, nom_keys = self.updateKeys(nom_tree, nom_keys)

        # add tree back to fixed root file
        fix_file[tree_name] = {key:nom_tree[key] for key in nom_keys}

        # save and close fixed root file
        print('Saved file: '+save_dir+'/'+in_name.split('.root')[0]+'_fixed_keys'+'.root')
        fix_file.close()


class truthPrep:
    """
    A class for prepping the MC truth values for TRecNet

        Methods:
            switchTTBarToHadLep: takes truth values determined by t and tbar and distinguishes them by thad and tlep
            addExtras: adds/computes missing truth information from MC including y, E, Pout, chi_tt, y_boost, and deltaPhi
            prepTruth: takes an input root files, preps the truth information in the tree (input parameter), and returns a new root file with the prepared truth info (and the other necessary branches from the input tree)
    """


    # Constructor
    def __init__(self):
        print("Creating truth prepper")


    def switchTTBarToHadLep(self, tree, keys):
        """
        Takes an input root tree and corresponding keys, and adds thad and tlep distinguisments for truth values depending on t and tbar

            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal tree

            Returns:
                tree (root tree): nominal tree with updated truth values
                keys: fixed keys for nominal tree
        """
        # Specify range of events
        range_events = range(len(tree["eventNumber"]))

        # First, clear all events that are not semi-leptonic
        sel_semi_lep = [(tree['MC_t_isHadronic'][evt] == 1 and tree['MC_tbar_isHadronic'][evt] == 0) or (tree['MC_t_isHadronic'][evt] == 0 and tree['MC_tbar_isHadronic'][evt] == 1) for evt in range_events]
        tree = tree[sel_semi_lep]

        # update range_events
        range_events = range(len(tree["eventNumber"]))

        # Get truth information for tops and bottoms (after FSR)
        for v in ['pt','eta','phi','m']:
            for p in ['t', 'b']:
                for s in ['afterFSR', 'beforeFSR']:
                    # Hadronic tops
                    keys.append('MC_'+p+'had_'+s+'_'+v)
                    tree['MC_'+p+'had_'+s+'_'+v] = [tree['MC_'+p+'_'+s+'_'+v][evt] if tree['MC_t_isHadronic'][evt] else tree['MC_'+p+'bar_'+s+'_'+v][evt] for evt in range_events]

                    # Leptonic tops
                    keys.append('MC_'+p+'lep_'+s+'_'+v)
                    tree['MC_'+p+'lep_'+s+'_'+v] = [tree['MC_'+p+'bar_'+s+'_'+v][evt] if tree['MC_t_isHadronic'][evt] else tree['MC_'+p+'_'+s+'_'+v][evt] for evt in range_events]

        # Get truth information for Ws and bottoms
        for v in ['pt','eta','phi','m']:
            for p in ['W', 'b']:
                # Hadronic tops
                keys.append('MC_'+p+'_from_thad_'+v)
                tree['MC_'+p+'_from_thad_'+v] = [tree['MC_'+p+'_from_t_'+v][evt] if tree['MC_t_isHadronic'][evt] else tree['MC_'+p+'_from_tbar_'+v][evt] for evt in range_events]

                # Leptonic tops
                keys.append('MC_'+p+'_from_tlep_'+v)
                tree['MC_'+p+'_from_tlep_'+v] = [tree['MC_'+p+'_from_tbar_'+v][evt] if tree['MC_t_isHadronic'][evt] else tree['MC_'+p+'_from_t_'+v][evt] for evt in range_events]

        # Get truth information for W decays
        for v in ['pt','eta','phi','m', 'pdgId']:
            for p in ['Wdecay1', 'Wdecay2']:
                # Hadronic tops
                keys.append('MC_'+p+'_from_thad_'+v)
                tree['MC_'+p+'_from_thad_'+v] = [tree['MC_'+p+'_from_t_'+v][evt] if tree['MC_t_isHadronic'][evt] else tree['MC_'+p+'_from_tbar_'+v][evt] for evt in range_events]

                # Leptonic tops
                keys.append('MC_'+p+'_from_tlep_'+v)
                tree['MC_'+p+'_from_tlep_'+v] = [tree['MC_'+p+'_from_tbar_'+v][evt] if tree['MC_t_isHadronic'][evt] else tree['MC_'+p+'_from_t_'+v][evt] for evt in range_events]

        # Get truth information for b/bbar to bhad/blep parent pdgId
        for v in ['parent1', 'parent2']:
            # Hadronic tops
            keys.append('MC_bhad_'+v+'_pdgId')
            tree['MC_bhad_'+v+'_pdgId'] = [tree['MC_b_'+v+'_pdgId'][evt] if tree['MC_t_isHadronic'][evt] else tree['MC_bbar_'+v+'_pdgId'][evt] for evt in range_events]

            # Leptonic tops
            keys.append('MC_blep_'+v+'_pdgId')
            tree['MC_blep_'+v+'_pdgId'] = [tree['MC_bbar_'+v+'_pdgId'][evt] if tree['MC_t_isHadronic'][evt] else tree['MC_b_'+v+'_pdgId'][evt] for evt in range_events]

        return tree, keys
    

    def addExtras(self, tree, keys):
        """
        Takes an input root tree and corresponding keys, and adds/computes missing observable information (y, E, deltaPhi, chi_tt, y_boost, and HT_tt)

            Parameters:
                tree (root tree): nominal tree
                keys: keys for nominal tree

            Returns:
                tree (root tree): nominal tree with updated truth values
                keys: fixed keys for nominal tree
        """

        # store 4-vectors for t_had_afterFSR and t_lep_afterFSR
        t_had_vec_afterFSR = vector.array({"pt":tree['MC_thad_afterFSR_pt'],"eta":tree['MC_thad_afterFSR_eta'],"phi":tree['MC_thad_afterFSR_phi'],"m":tree['MC_thad_afterFSR_m']})
        t_lep_vec_afterFSR = vector.array({"pt":tree['MC_tlep_afterFSR_pt'],"eta":tree['MC_tlep_afterFSR_eta'],"phi":tree['MC_tlep_afterFSR_phi'],"m":tree['MC_tlep_afterFSR_m']})

        # add keys, and rapidity and E to tree for thad and tlep (after FSR)
        keys.append('MC_thad_afterFSR_y')
        keys.append('MC_thad_afterFSR_E')
        tree['MC_thad_afterFSR_y'] = t_had_vec_afterFSR.rapidity
        tree['MC_thad_afterFSR_E'] = t_had_vec_afterFSR.E

        keys.append('MC_tlep_afterFSR_y')
        keys.append('MC_tlep_afterFSR_E')
        tree['MC_tlep_afterFSR_y'] = t_lep_vec_afterFSR.rapidity
        tree['MC_tlep_afterFSR_E'] = t_lep_vec_afterFSR.E

        # add deltaPhi to keys and tree
        keys.append('MC_deltaPhi_tt')
        tree['MC_deltaPhi_tt'] = np.absolute(t_had_vec_afterFSR.deltaphi(t_lep_vec_afterFSR))

        # add HT_tt to keys and tree
        keys.append('MC_HT_tt')
        tree['MC_HT_tt'] = np.add(t_had_vec_afterFSR.pt,t_lep_vec_afterFSR.pt) # scalar sum of t and tbar pTs

        # add y_boost to keys and tree
        keys.append('MC_y_boost')
        tree['MC_y_boost'] = 0.5 * np.add(t_had_vec_afterFSR.rapidity, t_lep_vec_afterFSR.rapidity)

        # add chi_tt to keys and tree
        keys.append('MC_chi_tt')
        tree['MC_chi_tt'] = np.exp(np.absolute(np.subtract(t_had_vec_afterFSR.rapidity, -1*t_lep_vec_afterFSR.rapidity)))

        # add pout for t and tbar
        keys.append('MC_thad_Pout')
        keys.append('MC_tlep_Pout')
        tree['MC_thad_Pout'] = t_had_vec_afterFSR.dot(t_lep_vec_afterFSR.cross(vector.obj(x=0,y=0, z=1)).unit())
        tree['MC_tlep_Pout'] = t_lep_vec_afterFSR.dot(t_had_vec_afterFSR.cross(vector.obj(x=0,y=0, z=1)).unit())

        # store 4-vector for ttbar_afterFSR
        ttbar_vec_afterFSR = vector.array({"pt":tree['MC_ttbar_afterFSR_pt'],"eta":tree['MC_ttbar_afterFSR_eta'],"phi":tree['MC_ttbar_afterFSR_phi'],"m":tree['MC_ttbar_afterFSR_m']})

        # add keys, and rapidity and E to tree for ttbar (after FSR)
        keys.append('MC_ttbar_afterFSR_y')
        keys.append('MC_ttbar_afterFSR_E')
        tree['MC_ttbar_afterFSR_y'] = ttbar_vec_afterFSR.rapidity
        tree['MC_ttbar_afterFSR_E'] = ttbar_vec_afterFSR.E

        return tree, keys



    def prepTruth(self, root_file, save_dir, tree_name):
        """
        Prepares the truth data for use in TRecNet by adding distinguishments between thad and tlep truth data and adding truth data not found in MC data
        
            Parameters:
                root_file: the full path to the input root file
                save_dir: the path to the desired save directory
                tree_name: the name of the nominal tree in the input root file (i.e. nominal_Loose)

        """
        in_name = os.path.split(root_file)[1]
        og_file = uproot.open(root_file)
        fix_file = uproot.recreate(save_dir+'/'+in_name.split('.root')[0]+'_fixed_truths'+'.root')

        # save keys and trees from original file
        nom_tree = og_file[tree_name].arrays()
        nom_keys = og_file[tree_name].keys()
        
        # close orginal file
        og_file.close()

        # update keys
        print('Converting truth values from t/tbar to thad/tlep')
        nom_tree, nom_keys = self.switchTTBarToHadLep(nom_tree, nom_keys)

        # add deltaPhi, HT, and chi for tt, and y_boost
        print('Computing remaining truth values')
        nom_tree, nom_keys = self.addExtras(nom_tree, nom_keys)

        # add tree back to fixed root file
        fix_file[tree_name] = {key:nom_tree[key] for key in nom_keys}

        # save and close fixed root file
        print('Saved file: '+save_dir+'/'+in_name.split('.root')[0]+'_fixed_keys'+'.root')
        fix_file.close()
        
        
        

# ---------- GET ARGUMENTS FROM COMMAND LINE ---------- #

# Create the main parser and subparsers
parser = ArgumentParser()
subparser = parser.add_subparsers(dest='function')
subparser.required = True
p_ttbbCut = subparser.add_parser('ttbbCut')
p_convertKeys = subparser.add_parser('convertKeys')
p_prepTruth = subparser.add_parser('prepTruth')

# Define arguments for ttbbCut
p_ttbbCut.add_argument('--root_file', help='input root file including path', required=True)
p_ttbbCut.add_argument('--save_dir', help='path to save directory', required=True)
p_ttbbCut.add_argument('--tree_name', help='name of tree', required=True)
p_ttbbCut.add_argument('--num_b_tags', help='minimum number of b tags per event', type=int,default=3)
p_ttbbCut.add_argument('--n_jets_min', help='minimum number of jets per event', type=int,default=6)
p_ttbbCut.add_argument('--DL1r_op_point', help='operating point for DL1r b-tagging network', type=int, default=3) # default: 3 -> 70 percent operating point

# Define arguments for convertKeys
p_convertKeys.add_argument('--root_file', help='input root file including path', required=True)
p_convertKeys.add_argument('--save_dir', help='path to save directory', required=True)
p_convertKeys.add_argument('--tree_name', help='name of tree', required=True)

# Define arguments for prepTruth
p_prepTruth.add_argument('--root_file', help='input root file including path', required=True)
p_prepTruth.add_argument('--save_dir', help='path to save directory', required=True)
p_prepTruth.add_argument('--tree_name', help='name of tree', required=True)


# Main script
args = parser.parse_args()
if args.function == 'ttbbCut':
    cutter = ttbbCutter()
    cutter.ttbbCut(args.root_file, args.save_dir, args.tree_name, args.num_b_tags, args.n_jets_min, args.DL1r_op_point)
elif args.function == 'convertKeys':
    converter = keyConverter()
    converter.convertKeys(args.root_file, args.save_dir, args.tree_name)
elif args.function == 'prepTruth':
    prepper = truthPrep()
    prepper.prepTruth(args.root_file, args.save_dir, args.tree_name)
else:
    print('Invalid function type.')