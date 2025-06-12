######################################################################
#                                                                    #
#  JetMatcher.py                                                     #
#  Author: Jenna Chisholm                                            #
#  Updated: Jun.5/25                                                 #
#                                                                    #
#  A class for matching ttbar decay products to reco level jets.     #    
#  The intention is to identify which jets are direct products of    #
#  ttbbar, and use this flag to help train TRecNet.                  #
#                                                                    #
#  Thoughts for improvements: Use config to get variable names.      #
#                                                                    #
######################################################################

import uproot
import numpy as np
import vector
import os
from argparse import ArgumentParser
from Util import *

# Define some helpful pdgid ranges
light_quarks = list(range(-4,0)) + list(range(1,5))
b_quarks = [-5,5]

class JetMatcher:
    """ 
    A class for matching ttbar decay products to reco level jets.

        Methods:
            pruneTrees: Removes events we can't use from the trees (i.e. events with nan y and non-semileptonic events)
            getMatches: Matches ttbar decay products to reco-level jets.
            appendJetMatches: Gets the jet match tags and creates a new root file from the old one, including these new tags.

    """

    def __init__(self):
        print("Creating jetMatcher.")


    def getMatches(self,nom_tree, dR_cut, allowDoubleMatch, var_conf):
        """
        Matches ttbar decay products to reco-level jets.

            Parameters:
                nom_tree (root tree): Nominal tree from the root file.
                dR_cut (float): A threshold which the dR for all matches must be below.
                allowDoubleMatch (bool): Whether or not two or more decay products are allowed to be matched to the same jet.
                var_conf (str): Name (including path) of the variable names config file.

            Returns:
                isttbarJet (jagged array of bools): Tags for each jet in each event, where 0 means it was not matched to something, and 1 means it was.
                match_info (ndarray): Array of match info for each decay product in all events (form: [event index, decay particle, matched jet, (absolute) jet pdgid, dR for the match, fractional delta pt for the match]). 
        """

        # Create a list to save all the matched labels in
        isttbarJet = []
        match_info = []     # Just gonna be one long list my dude
        
        # Get the necessary ntuple observable names
        str_bh_pt, str_bh_eta, str_bh_phi, str_bh_m = getObservableNames(var_conf,'bh_pt','bh_eta','bh_phi','bh_m')
        str_bl_pt, str_bl_eta, str_bl_phi, str_bl_m = getObservableNames(var_conf,'bl_pt','bl_eta','bl_phi','bl_m')
        str_wh_decay1_pt, str_wh_decay1_eta, str_wh_decay1_phi, str_wh_decay1_m, str_wh_decay1_pdgid = getObservableNames(var_conf,'wh_decay1_pt','wh_decay1_eta','wh_decay1_phi','wh_decay1_m','wh_decay1_pdgid')
        str_wh_decay2_pt, str_wh_decay2_eta, str_wh_decay2_phi, str_wh_decay2_m, str_wh_decay2_pdgid = getObservableNames(var_conf,'wh_decay2_pt','wh_decay2_eta','wh_decay2_phi','wh_decay2_m','wh_decay2_pdgid')
        str_jet_pt, str_jet_eta, str_jet_phi, str_jet_e, str_jet_partonLabel, str_jet_n = getObservableNames(var_conf,'jet_pt','jet_eta','jet_phi','jet_e','jet_partonLabel','jet_n')

        # Calculate particle vectors
        b_from_thad_vec = vector.array({"pt":nom_tree[str_bh_pt],"eta":nom_tree[str_bh_eta],"phi":nom_tree[str_bh_phi],"m":nom_tree[str_bh_m]})
        b_from_tlep_vec = vector.array({"pt":nom_tree[str_bl_pt],"eta":nom_tree[str_bl_eta],"phi":nom_tree[str_bl_phi],"m":nom_tree[str_bl_m]})
        Wdecay1_from_thad_vec = vector.array({"pt":nom_tree[str_wh_decay1_pt],"eta":nom_tree[str_wh_decay1_eta],"phi":nom_tree[str_wh_decay1_phi],"m":nom_tree[str_wh_decay1_m]})
        Wdecay2_from_thad_vec = vector.array({"pt":nom_tree[str_wh_decay2_pt],"eta":nom_tree[str_wh_decay2_eta],"phi":nom_tree[str_wh_decay2_phi],"m":nom_tree[str_wh_decay2_m]})

        # Need to go through event by event :(
        num_events = len(nom_tree['eventNumber'])
        for i in range(num_events):

            # Get the number of jets in this event
            n_jets = int(nom_tree[str_jet_n][i])

            # Calculate the jet vectors for this event, as well as get the parton the jet originated from
            jet_vectors = vector.array({"pt":nom_tree[str_jet_pt][i],"eta":nom_tree[str_jet_eta][i],"phi":nom_tree[str_jet_phi][i],"E":nom_tree[str_jet_e][i]})
            jet_quarks = np.array(nom_tree[str_jet_partonLabel][i])

            # Create a set of jet labels (0 to jn-1) and matched labels (items will be moved from jet labels into matched labels as they're matched, if double matching is not allowed)
            jet_labels = list(range(n_jets))
            event_matched_labels = []

            # Get vectors of all the hadronic decay products, and calculate their dRs and fractional delta pts with the jets
            particle_dict = {'b_from_thad':{'dRs':jet_vectors.deltaR(b_from_thad_vec[i]),'frac_delta_pts':((b_from_thad_vec[i].pt - jet_vectors.pt)/b_from_thad_vec[i].pt)},
                            'b_from_tlep':{'dRs':jet_vectors.deltaR(b_from_tlep_vec[i]),'frac_delta_pts':((b_from_tlep_vec[i].pt - jet_vectors.pt)/b_from_tlep_vec[i].pt)},
                            'Wdecay1_from_thad':{'dRs':jet_vectors.deltaR(Wdecay1_from_thad_vec[i]),'frac_delta_pts':((Wdecay1_from_thad_vec[i].pt - jet_vectors.pt)/Wdecay1_from_thad_vec[i].pt)},
                            'Wdecay2_from_thad':{'dRs':jet_vectors.deltaR(Wdecay2_from_thad_vec[i]),'frac_delta_pts':((Wdecay2_from_thad_vec[i].pt - jet_vectors.pt)/Wdecay2_from_thad_vec[i].pt)},}


            # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
            for par in particle_dict:

                # Get the previously calculated information
                dRs, pts = np.array(particle_dict[par]['dRs']), np.array(particle_dict[par]['frac_delta_pts'])

                # First limit our pool of choices to those with dR <= dR_cut
                sel = np.array(dRs<=dR_cut)

                # If not allowing double matching, also remove jets that have already been matched
                if not allowDoubleMatch:
                    sel = sel * np.array([True if j in jet_labels else False for j in range(n_jets)])

                # Want it to be the same quark type
                if 'Wdecay1' in par:
                    pdgid = abs(nom_tree[str_wh_decay1_pdgid][i])
                    sel = sel * np.array([True if q==pdgid else False for q in jet_quarks])
                elif 'Wdecay2' in par:
                    pdgid = abs(nom_tree[str_wh_decay2_pdgid][i])
                    sel = sel * np.array([True if q==pdgid else False for q in jet_quarks])
                else:
                    sel = sel * np.array([True if q in b_quarks else False for q in jet_quarks])

                # Make these selections
                dRs_afterCuts = dRs[sel]
                pts_afterCuts = pts[sel]

                # If there is nothing left after the above selections, move on to the next particle
                if len(dRs_afterCuts)==0:
                    continue

                # Else if there are options with a decent fractional delta pt, prioritize these
                elif True in list(abs(pts_afterCuts)<=1):
                    dRs_afterCuts = dRs_afterCuts[abs(pts_afterCuts)<=1]
                    pts_afterCuts = pts_afterCuts[abs(pts_afterCuts)<=1]

                # Get the minimum dR from the cut down list, and find the jet and its truth parton and frac delta pt associated
                best_dR = np.min(dRs_afterCuts)
                best_jet = np.where(dRs==best_dR)[0][0]
                best_jet_truth = jet_quarks[best_jet]
                best_frac_delta_pt = pts[best_jet]

                # Save the best match we ended up with, and remove that jet from the list (so that it's not doubly assigned)
                event_matched_labels.append(best_jet)   # Save the matched jet label
                if not allowDoubleMatch: jet_labels.remove(best_jet) 

                # Also save the fractional delta pt between the particle and best jet
                match_info.append([i,par,best_jet,best_jet_truth,best_dR,best_frac_delta_pt])


            # Get list of bools for whether or not jets are ttbar, and then append to array
            if len(event_matched_labels)==0:
                eventJetBools = [0 for j in range(n_jets)]
            else:
                eventJetBools = [1 if j in event_matched_labels else 0 for j in range(n_jets)]
            isttbarJet.append(eventJetBools)


            # Print every once in a while so we know there's progress
            if (i+1)%100000==0 or i==num_events-1:
                print ('Jet Events Processed: '+str(i+1))


        return isttbarJet, match_info


    def appendJetMatches(self,input_file,save_dir,var_conf,dR_cut,allowDoubleMatching):
        """
        Gets the jet match tags and creates a new root file from the old one, including these new tags.

            Parameters:
                input_file (str): Name (including path) of the root file you'd like to add the jet matches to.
                save_dir (str): Desired directory to save the output root file in.
                var_conf (str): Name (including path) of the config file for the variable names.
                dR_cut (float): A threshold which the dR for all matches must be below.
                allowDoubleMatch (bool): Whether or not two or more decay products are allowed to be matched to the same jet.

            Returns:
                Creates a new root file that includes the systematic uncertainty trees, as well as the nominal tree with the new jet match tags included as 'jet_isTruth'.
        """

        # Separate input file name and its path
        #in_path = os.path.split(input_file)[0]
        in_name = os.path.split(input_file)[1]

        # Just need this little string for file saving purposes
        match_tag = '_jetMatch'+str(dR_cut).replace('.','')

        # Open the original file and a new file to write to
        print('Opening root file ...')
        og_file = uproot.open(input_file)

        # Get the reco and parton trees from the original file
        nom_name, up_name, down_name = getBranchNames(var_conf)
        down_tree = og_file[down_name].arrays()
        up_tree = og_file[up_name].arrays()
        nom_tree = og_file[nom_name].arrays()

        # Save the keys for later
        down_keys = og_file[down_name].keys()
        up_keys = og_file[up_name].keys()
        nom_keys = og_file[nom_name].keys()

        # Close the original file
        og_file.close()

        # Remove events from the trees that we can't use
        #print('Pruning trees ...')
        #nom_tree = self.pruneTree(nom_tree)  # MOVED TO EVENTREMOVER

        # Get the jet matches
        print('Matching jets ...')
        isttbarJet, matching_info = self.getMatches(nom_tree, dR_cut, allowDoubleMatching)

        # Save the truth tags to the reco tree
        nom_tree['jet_isTruth'] = isttbarJet
        nom_keys.append('jet_isTruth')

        # Write the trees to the file
        print('Writing trees to new file...')
        new_file = uproot.recreate(save_dir+'/'+in_name.split('.root')[0]+match_tag+'.root')
        new_file[down_name] = {key:down_tree[key] for key in down_keys}
        new_file[up_name] = {key:up_tree[key] for key in up_keys}
        new_file[nom_name] = {key:nom_tree[key] for key in nom_keys}

        print('Saved file: '+save_dir+'/'+in_name.split('.root')[0]+match_tag+'.root')

        # Close new file
        new_file.close()

        # Save the matching info separately, since it's a weird shape, and save in its own folder
        if not os.path.exists(save_dir+'/matching_info'):
            os.mkdir(save_dir+'/matching_info')
        np.save(save_dir+'/matching_info/'+in_name.split('.root')[0]+match_tag,matching_info)
        print('Saved file: '+save_dir+'/matching_info/'+in_name.split('.root')[0]+match_tag+'.npy')
        
        
# ---------- GET ARGUMENTS FROM COMMAND LINE ---------- #      
        
# Create the main parser
parser = ArgumentParser()

# Define arguments for JetMatcher
parser.add_argument('--input',help='Input file (including path).',required=True)
parser.add_argument('--save_dir',help='Path for directory where file will be saved.',required=True)
parser.add_argument('--var_conf',help='Config file (including path) for names of variables.',required=True)
parser.add_argument('--dR_cut',help='Maximum dR for the cut on dR (default: 1.0).',type=float,default=1.0)
parser.add_argument('--allow_double_matching',help='Use this flag to allow double matching.',action='store_false')

# Parse the arguments and proceed with stuff
args = parser.parse_args()
matcher = JetMatcher()
matcher.appendJetMatches(args.input,args.save_dir,args.var_conf,args.dR_cut,args.allow_double_matching)

print('Done :)')