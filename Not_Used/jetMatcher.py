import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('GTK3Agg')   # need this one if you want plots displayed
import vector
import os


# Define some helpful ranges
had_range = list(range(1,9))+list(range(-8,0))                     # Quarks have pdgid 1 to 8 (negatives indicate antiparticles)
lep_range = list(range(11,19))+list(range(-18,10))                 # Leptons have pdgid 11 to 18
bos_range = list(range(21,26))+list(range(-25,20))+[9,37,-9,-37]   # Bosons have pdgid 21 to 25, and 9, and 37

light_quarks = list(range(-4,0)) + list(range(1,5))
b_quarks = [-5,5]




def pruneTrees(nom_tree):
    """
    Removes bad events from the trees (i.e. events with nan y, isDummy events, !isMatched events, !isTruthSemileptonic events)
    Input: parton level tree, reco level tree
    Output: parton level tree, reco level tree
    """

    # Don't want nan values
    for var in ['MC_thad_afterFSR_y','MC_tlep_afterFSR_y','MC_W_from_thad_y','MC_W_from_tlep_y']:
        sel = np.invert(np.isnan(nom_tree[var]))

    # Only want semi-leptonic events
    sel = sel*(nom_tree['semileptonicEvent']==1)

    # Make a cut on met_met
    #sel = sel*nom_tree['met_met']/1000 >= met_cut

    # Make these selections
    nom_tree = nom_tree[sel]


    return nom_tree



def getMatches(nom_tree, dR_cut, allowDoubleMatch):


    # Create a list to save all the matched labels in
    isttbarJet = []
    match_info = []     # Just gonna be one long list my dude


    # Calculate particle vectors
    b_from_thad_vec = vector.array({"pt":nom_tree['MC_b_from_thad_pt'],"eta":nom_tree['MC_b_from_thad_eta'],"phi":nom_tree['MC_b_from_thad_phi'],"m":nom_tree['MC_b_from_thad_m']})
    b_from_tlep_vec = vector.array({"pt":nom_tree['MC_b_from_tlep_pt'],"eta":nom_tree['MC_b_from_tlep_eta'],"phi":nom_tree['MC_b_from_tlep_phi'],"m":nom_tree['MC_b_from_tlep_m']})
    Wdecay1_from_thad_vec = vector.array({"pt":nom_tree['MC_Wdecay1_from_thad_pt'],"eta":nom_tree['MC_Wdecay1_from_thad_eta'],"phi":nom_tree['MC_Wdecay1_from_thad_phi'],"m":nom_tree['MC_Wdecay1_from_thad_m']})
    Wdecay2_from_thad_vec = vector.array({"pt":nom_tree['MC_Wdecay2_from_thad_pt'],"eta":nom_tree['MC_Wdecay2_from_thad_eta'],"phi":nom_tree['MC_Wdecay2_from_thad_phi'],"m":nom_tree['MC_Wdecay2_from_thad_m']})


    # Need to go through event by event :(
    num_events = len(nom_tree['eventNumber'])
    for i in range(num_events):

        # Get the number of jets in this event
        n_jets = int(nom_tree['jet_n'][i])

        # Calculate the jet vectors for this event, as well as get the parton the jet originated from
        jet_vectors = vector.array({"pt":nom_tree['jet_pt'][i],"eta":nom_tree['jet_eta'][i],"phi":nom_tree['jet_phi'][i],"E":nom_tree['jet_e'][i]})
        jet_quarks = np.array(nom_tree['jet_truthPartonLabel'][i])

        # Create a set of jet labels (0 to jn-1) and matched labels (items will be moved from jet labels into matched labels as they're matched, if double matching is not allowed)
        jet_labels = list(range(n_jets))
        event_matched_labels = []
        #event_match_info = []

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
            if par[0]=='W':
                pdgid = abs(nom_tree['MC_'+par+'_pdgId'][i])
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

        #all_match_info.append(event_match_info)

        # Print every once in a while so we know there's progress
        if (i+1)%100000==0 or i==num_events-1:
            print ('Jet Events Processed: '+str(i+1))


    return isttbarJet, match_info



def addMatchesToFile(directory,file,dR_cut,allowDoubleMatching=True):


    # Just need this little string for file saving purposes
    match_tag = '_jetMatch'+str(dR_cut).replace('.','')

    # Open the original file and a new file to write to
    print('Opening root files ...')
    og_file = uproot.open(file.path)
    fix_file = uproot.recreate(file.path.split('.root')[0]+match_tag+'.root')

    # Get the reco and parton trees from the original file
    down_tree = og_file['CategoryReduction_JET_Pileup_RhoTopology__1down'].arrays()
    up_tree = og_file['CategoryReduction_JET_Pileup_RhoTopology__1up'].arrays()
    nom_tree = og_file['nominal'].arrays()

    # Save the keys for later
    down_keys = og_file['CategoryReduction_JET_Pileup_RhoTopology__1down'].keys()
    up_keys = og_file['CategoryReduction_JET_Pileup_RhoTopology__1up'].keys()
    nom_keys = og_file['nominal'].keys()

    # Close the original file
    og_file.close()

    # Remove events from the trees that we can't use
    print('Pruning trees ...')
    nom_tree = pruneTrees(nom_tree)

    # Get the jet matches
    print('Matching jets ...')
    isttbarJet, matching_info = getMatches(nom_tree, dR_cut, allowDoubleMatching)

    # Save the truth tags to the reco tree
    nom_tree['jet_isTruth'] = isttbarJet
    nom_keys.append('jet_isTruth')

    # Write the trees to the file
    print('Writing trees ...')
    fix_file['CategoryReduction_JET_Pileup_RhoTopology__1down'] = {key:down_tree[key] for key in down_keys}
    fix_file['CategoryReduction_JET_Pileup_RhoTopology__1up'] = {key:up_tree[key] for key in up_keys}
    fix_file['nominal'] = {key:nom_tree[key] for key in nom_keys}

    print('Saved file: '+file.path.split('.root')[0]+match_tag+'.root')

    # Close new file
    fix_file.close()

    # Save the matching info separately, since it's a weird shape
    np.save(file.path.split(file.name)[0]+'matching_info/matching_info_'+file.name.split('.root')[0]+match_tag,matching_info)
    print('Saved file: '+file.path.split(file.name)[0]+'matching_info/matching_info_'+file.name.split('.root')[0]+match_tag+'.npy')










# ----- MAIN CODE ----- #

directory = '/data/jchishol/mc1516_new'
 
# Iterate over files in that directory
for file in sorted(os.scandir(directory),key=lambda f: f.name):
    if file.is_file():
        print('On file ',file.name)
        addMatchesToFile(directory,file,dR_cut=0.4,allowDoubleMatching=True)