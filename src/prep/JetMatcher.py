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
#  Thoughts for improvements:                                        #
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
            getMatches: Matches ttbar decay products to reco-level jets.

    """

    def __init__(self):
        print("Creating jetMatcher.")
        
    def getParticleVecs(self, nom_tree, var_conf, mode):
        
        # Get the necessary ntuple observable names
        str_bh_pt, str_bh_eta, str_bh_phi, str_bh_m = getObservableNames(var_conf,'bh_pt','bh_eta','bh_phi','bh_m')
        str_bl_pt, str_bl_eta, str_bl_phi, str_bl_m = getObservableNames(var_conf,'bl_pt','bl_eta','bl_phi','bl_m')
        str_wh_decay1_pt, str_wh_decay1_eta, str_wh_decay1_phi, str_wh_decay1_m = getObservableNames(var_conf,'wh_decay1_pt','wh_decay1_eta','wh_decay1_phi','wh_decay1_m')
        str_wh_decay2_pt, str_wh_decay2_eta, str_wh_decay2_phi, str_wh_decay2_m = getObservableNames(var_conf,'wh_decay2_pt','wh_decay2_eta','wh_decay2_phi','wh_decay2_m')
        
        # Calculate particle vectors and add them to the dic
        b_from_thad_vec = vector.array({"pt":nom_tree[str_bh_pt],"eta":nom_tree[str_bh_eta],"phi":nom_tree[str_bh_phi],"m":nom_tree[str_bh_m]})
        b_from_tlep_vec = vector.array({"pt":nom_tree[str_bl_pt],"eta":nom_tree[str_bl_eta],"phi":nom_tree[str_bl_phi],"m":nom_tree[str_bl_m]})
        Wdecay1_from_thad_vec = vector.array({"pt":nom_tree[str_wh_decay1_pt],"eta":nom_tree[str_wh_decay1_eta],"phi":nom_tree[str_wh_decay1_phi],"m":nom_tree[str_wh_decay1_m]})
        Wdecay2_from_thad_vec = vector.array({"pt":nom_tree[str_wh_decay2_pt],"eta":nom_tree[str_wh_decay2_eta],"phi":nom_tree[str_wh_decay2_phi],"m":nom_tree[str_wh_decay2_m]})
        particle_vecs = {'b_from_thad_vec': b_from_thad_vec, 'b_from_tlep_vec': b_from_tlep_vec, 'Wdecay1_from_thad_vec': Wdecay1_from_thad_vec, 'Wdecay2_from_thad_vec': Wdecay2_from_thad_vec}
        
        # Do the same for ttbb cases, if that is the mode
        if mode=='ttbar_bbbar':
            str_b_pt, str_b_eta, str_b_phi, str_b_m = getObservableNames(var_conf,'b_pt','b_eta','b_phi','b_m')
            str_bbar_pt, str_bbar_eta, str_bbar_phi, str_bbar_m = getObservableNames(var_conf,'bbar_pt','bbar_eta','bbar_phi','bbar_m')
            b_vec = vector.array({"pt":nom_tree[str_b_pt],"eta":nom_tree[str_b_eta],"phi":nom_tree[str_b_phi],"m":nom_tree[str_b_m]})
            bbar_vec = vector.array({"pt":nom_tree[str_bbar_pt],"eta":nom_tree[str_bbar_eta],"phi":nom_tree[str_bbar_phi],"m":nom_tree[str_bbar_m]})
            particle_vecs.update({'b_vec':b_vec, 'bbar_vec':bbar_vec})
        elif mode=='ttbar_b1b2':
            str_b1_pt, str_b1_eta, str_b1_phi, str_b1_m = getObservableNames(var_conf,'b1_pt','b1_eta','b1_phi','b1_m')
            str_b2_pt, str_b2_eta, str_b2_phi, str_b2_m = getObservableNames(var_conf,'b2_pt','b2_eta','b2_phi','b2_m')
            b1_vec = vector.array({"pt":nom_tree[str_b1_pt],"eta":nom_tree[str_b1_eta],"phi":nom_tree[str_b1_phi],"m":nom_tree[str_b1_m]})
            b2_vec = vector.array({"pt":nom_tree[str_b2_pt],"eta":nom_tree[str_b2_eta],"phi":nom_tree[str_b2_phi],"m":nom_tree[str_b2_m]})
            particle_vecs.update({'b1_vec':b1_vec, 'b2_vec':b2_vec})
            
        return particle_vecs
            
        


    def getMatches(self,nom_tree, dR_cut, allowDoubleMatch, var_conf, mode):
        """
        Matches ttbar decay products to reco-level jets.

            Parameters:
                nom_tree (root tree): Nominal tree from the root file.
                dR_cut (float): A threshold which the dR for all matches must be below.
                allowDoubleMatch (bool): Whether or not two or more decay products are allowed to be matched to the same jet.
                var_conf (str): Name (including path) of the variable names config file.
                mode (str): 'ttbar', 'ttbar_bbbar', or 'ttbar_b1b2'

            Returns:
                isttbarJet (jagged array of bools): Tags for each jet in each event, where 0 means it was not matched to something, and 1 means it was.
                match_info (ndarray): Array of match info for each decay product in all events (form: [event index, decay particle, matched jet, (absolute) jet pdgid, dR for the match, fractional delta pt for the match]). 
        """

        # Create a list to save all the matched labels in
        isttbarJet = []
        match_info = []     # Just gonna be one long list my dude
        
        # Get list of MC particle vectors
        particle_vecs = self.getParticleVecs(nom_tree, var_conf, mode)
        
        # Get ntuple jet and pdgid observable names
        str_jet_pt, str_jet_eta, str_jet_phi, str_jet_e, str_jet_partonLabel, str_jet_n = getObservableNames(var_conf,'jet_pt','jet_eta','jet_phi','jet_e','jet_partonLabel','jet_n')
        str_wh_decay1_pdgid = getObservableName(var_conf,'wh_decay1_pdgid')
        str_wh_decay2_pdgid = getObservableName(var_conf,'wh_decay2_pdgid')

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

            # Calculate dRs and fractional delta pts of all MC particles with the reco jets
            particle_dict = {}
            if mode=='ttbar_bbbar':
                particle_dict.update({'b':{'dRs':jet_vectors.deltaR(particle_vecs['b_vec'][i]),'frac_delta_pts':((particle_vecs['b_vec'][i].pt - jet_vectors.pt)/particle_vecs['b_vec'][i].pt)},
                                      'bbar':{'dRs':jet_vectors.deltaR(particle_vecs['bbar_vec'][i]),'frac_delta_pts':((particle_vecs['bbar_vec'][i].pt - jet_vectors.pt)/particle_vecs['bbar_vec'][i].pt)}})
            elif mode=='ttbar_b1b2':
                particle_dict.update({'b1':{'dRs':jet_vectors.deltaR(particle_vecs['b1_vec'][i]),'frac_delta_pts':((particle_vecs['b1_vec'][i].pt - jet_vectors.pt)/particle_vecs['b1_vec'][i].pt)},
                                      'b2':{'dRs':jet_vectors.deltaR(particle_vecs['b2_vec'][i]),'frac_delta_pts':((particle_vecs['b2_vec'][i].pt - jet_vectors.pt)/particle_vecs['b2_vec'][i].pt)}})
            
            particle_dict.update({'b_from_thad':{'dRs':jet_vectors.deltaR(particle_vecs['b_from_thad_vec'][i]),'frac_delta_pts':((particle_vecs['b_from_thad_vec'][i].pt - jet_vectors.pt)/particle_vecs['b_from_thad_vec'][i].pt)},
                            'b_from_tlep':{'dRs':jet_vectors.deltaR(particle_vecs['b_from_tlep_vec'][i]),'frac_delta_pts':((particle_vecs['b_from_tlep_vec'][i].pt - jet_vectors.pt)/particle_vecs['b_from_tlep_vec'][i].pt)},
                            'Wdecay1_from_thad':{'dRs':jet_vectors.deltaR(particle_vecs['Wdecay1_from_thad_vec'][i]),'frac_delta_pts':((particle_vecs['Wdecay1_from_thad_vec'][i].pt - jet_vectors.pt)/particle_vecs['Wdecay1_from_thad_vec'][i].pt)},
                            'Wdecay2_from_thad':{'dRs':jet_vectors.deltaR(particle_vecs['Wdecay2_from_thad_vec'][i]),'frac_delta_pts':((particle_vecs['Wdecay2_from_thad_vec'][i].pt - jet_vectors.pt)/particle_vecs['Wdecay2_from_thad_vec'][i].pt)}})
            

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
    
    