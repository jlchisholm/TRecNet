import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('GTK3Agg')   # need this one if you want plots displayed
import vector


# Define some helpful ranges
had_range = list(range(1,9))+list(range(-8,0))                     # Quarks have pdgid 1 to 8 (negatives indicate antiparticles)
lep_range = list(range(11,19))+list(range(-18,10))                 # Leptons have pdgid 11 to 18
bos_range = list(range(21,26))+list(range(-25,20))+[9,37,-9,-37]   # Bosons have pdgid 21 to 25, and 9, and 37




# ---------- FUNCTION FOR REMOVING EVENTS WE CAN'T USE ---------- #

def pruneTrees(parton_tree,reco_tree):
    """
    Removes bad events from the trees (i.e. events with nan y, isDummy events, !isMatched events, !isTruthSemileptonic events)
    Input: parton level tree, reco level tree
    Output: parton level tree, reco level tree
    """

    # Remove events with nan y values
    for var in ['MC_thad_afterFSR_y','MC_tlep_afterFSR_y','MC_Wdecay1_from_t_afterFSR_y','MC_Wdecay2_from_t_afterFSR_y','MC_Wdecay1_from_tbar_afterFSR_y','MC_Wdecay2_from_tbar_afterFSR_y']:
        sel = np.invert(np.isnan(parton_tree[var]))
        parton_tree = parton_tree[sel]
        reco_tree = reco_tree[sel]

    # Remove isDummy events and !isMatched and isTruthSemileptonic events
    sel = reco_tree['isMatched']==1
    parton_tree = parton_tree[sel]
    reco_tree = reco_tree[sel]
    sel = parton_tree['isDummy']==0
    parton_tree = parton_tree[sel]
    reco_tree = reco_tree[sel]
    sel = reco_tree['isTruthSemileptonic']==1
    parton_tree = parton_tree[sel]
    reco_tree = reco_tree[sel]

    return parton_tree, reco_tree

# ---------- FUNCTION FOR GETTING W AND B VARIABLES ---------- #

def appendWandB(parton_tree):
    """
    Calculates the W and b variables and puts them in the parton level tree
    Input: parton level tree
    Output: parton level tree
    """

    # For both the t and tbar sides ...
    for t in ['t','tbar']:

        print('Calculating 4-vectors on '+t+' side ...')

        # Create lists for the variables
        W_variables = []
        b_variables = []

        # Create the W decay vectors
        # ('Wdecay2_from_t_eta' is full of zeroes, so we need to approximate with y for now)
        Wdecay1_vec = vector.array({"pt":parton_tree['MC_Wdecay1_from_'+t+'_afterFSR_pt'],"phi":parton_tree['MC_Wdecay1_from_'+t+'_afterFSR_phi'],"eta":parton_tree['MC_Wdecay1_from_'+t+'_afterFSR_eta'],"mass":parton_tree['MC_Wdecay1_from_'+t+'_afterFSR_m']})
        if t=='t':
            Wdecay2_vec = vector.array({"pt":parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_pt'],"phi":parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_phi'],"eta":parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_y'],"mass":parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_m']})
        else:
            Wdecay2_vec = vector.array({"pt":parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_pt'],"phi":parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_phi'],"eta":parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_eta'],"mass":parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_m']})

        # Add decays together to get W vector
        W_vec = Wdecay1_vec + Wdecay2_vec

        # Get t or tbar vector
        t_vec = vector.array({"pt":parton_tree['MC_'+t+'_afterFSR_pt'],"phi":parton_tree['MC_'+t+'_afterFSR_phi'],"eta":parton_tree['MC_'+t+'_afterFSR_eta'],"mass":parton_tree['MC_'+t+'_afterFSR_m']})

        # Subtract W from t to get b quark vector
        b_vec = t_vec - W_vec

        # Put values in tree
        print('Appending values to tree ...')

        parton_tree['MC_W_from_'+t+'_afterFSR_pt'] = W_vec.pt
        parton_tree['MC_W_from_'+t+'_afterFSR_eta'] = W_vec.eta
        parton_tree['MC_W_from_'+t+'_afterFSR_y'] = W_vec.rapidity
        parton_tree['MC_W_from_'+t+'_afterFSR_phi'] = W_vec.phi
        parton_tree['MC_W_from_'+t+'_afterFSR_m'] = W_vec.m
        parton_tree['MC_W_from_'+t+'_afterFSR_E'] = W_vec.E

        parton_tree['MC_b_from_'+t+'_afterFSR_pt'] = b_vec.pt
        parton_tree['MC_b_from_'+t+'_afterFSR_eta'] = b_vec.eta
        parton_tree['MC_b_from_'+t+'_afterFSR_y'] = b_vec.rapidity
        parton_tree['MC_b_from_'+t+'_afterFSR_phi'] = b_vec.phi
        parton_tree['MC_b_from_'+t+'_afterFSR_m'] = b_vec.m
        parton_tree['MC_b_from_'+t+'_afterFSR_E'] = b_vec.E

    return parton_tree


# ---------- FUNCTION FOR GETTING MATCHING JETS AND QUARKS ---------- #

def getMatches(parton_tree, reco_tree, dR_cut, allowDoubleMatching):
    """
    Matches each of the four hadronic decay products to a jet.
    Input: parton level tree, reco level tree, cut for dR, and boolean for whether you want to allow double matching
    Output: array of booleans for whether the jets in an event were matched, and (not very structured) array with info about the matches
    """

    # Create a list to save all the matched labels in
    isttbarJet = []
    useful_stuff = []   # This list will not be separated by events or by jets, it's just one big long list of useful things

    # Calculate particle vectors
    b_from_t_vec = vector.array({"pt":parton_tree['MC_b_from_t_afterFSR_pt'],"eta":parton_tree['MC_b_from_t_afterFSR_eta'],"phi":parton_tree['MC_b_from_t_afterFSR_phi'],"E":parton_tree['MC_b_from_t_afterFSR_E']})
    b_from_tbar_vec = vector.array({"pt":parton_tree['MC_b_from_tbar_afterFSR_pt'],"eta":parton_tree['MC_b_from_tbar_afterFSR_eta'],"phi":parton_tree['MC_b_from_tbar_afterFSR_phi'],"E":parton_tree['MC_b_from_tbar_afterFSR_E']})
    Wdecay1_from_t_vec = vector.array({"pt":parton_tree['MC_Wdecay1_from_t_afterFSR_pt'],"eta":parton_tree['MC_Wdecay1_from_t_afterFSR_eta'],"phi":parton_tree['MC_Wdecay1_from_t_afterFSR_phi'],"E":parton_tree['MC_Wdecay1_from_t_afterFSR_E']})
    Wdecay2_from_t_vec = vector.array({"pt":parton_tree['MC_Wdecay2_from_t_afterFSR_pt'],"eta":parton_tree['MC_Wdecay2_from_t_afterFSR_y'],"phi":parton_tree['MC_Wdecay2_from_t_afterFSR_phi'],"E":parton_tree['MC_Wdecay2_from_t_afterFSR_E']})   # Approx eta with y for now, since eta empty
    Wdecay1_from_tbar_vec = vector.array({"pt":parton_tree['MC_Wdecay1_from_tbar_afterFSR_pt'],"eta":parton_tree['MC_Wdecay1_from_tbar_afterFSR_eta'],"phi":parton_tree['MC_Wdecay1_from_tbar_afterFSR_phi'],"E":parton_tree['MC_Wdecay1_from_tbar_afterFSR_E']})
    Wdecay2_from_tbar_vec = vector.array({"pt":parton_tree['MC_Wdecay2_from_tbar_afterFSR_pt'],"eta":parton_tree['MC_Wdecay2_from_tbar_afterFSR_eta'],"phi":parton_tree['MC_Wdecay2_from_tbar_afterFSR_phi'],"E":parton_tree['MC_Wdecay2_from_tbar_afterFSR_E']})


    # Need to go through event by event :(
    num_events = len(reco_tree['eventNumber'])
    for i in range(num_events):

        # Get the number of jets in this event
        n_jets = int(reco_tree['jet_n'][i])

        # Calculate the jet vectors for this event, as well as get the btags
        jet_vectors = vector.array({"pt":reco_tree['jet_pt'][i],"eta":reco_tree['jet_eta'][i],"phi":reco_tree['jet_phi'][i],"E":reco_tree['jet_e'][i]})
        jet_btags = np.array(reco_tree['jet_btagged'][i])

        # Create a set of jet labels (0 to jn-1) and matched labels (items will be moved from jet labels into matched labels as they're matched, if double matching is not allowed)
        jet_labels = list(range(n_jets))
        event_matched_labels = []

        # Get vectors of all the hadronic decay products, and calculate their dRs and fractional delta pts with the jets
        particle_dict = {'b_from_t':{'dRs':jet_vectors.deltaR(b_from_t_vec[i]),'frac_delta_pts':((b_from_t_vec[i].pt - jet_vectors.pt)/b_from_t_vec[i].pt)},'b_from_tbar':{'dRs':jet_vectors.deltaR(b_from_tbar_vec[i]),'frac_delta_pts':((b_from_tbar_vec[i].pt - jet_vectors.pt)/b_from_tbar_vec[i].pt)}}
        if parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][i] in had_range or parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][i] in had_range:
            particle_dict['Wdecay1_from_t'] = {'dRs':jet_vectors.deltaR(Wdecay1_from_t_vec[i]),'frac_delta_pts':((Wdecay1_from_t_vec[i].pt - jet_vectors.pt)/Wdecay1_from_t_vec[i].pt)}
            particle_dict['Wdecay2_from_t'] = {'dRs':jet_vectors.deltaR(Wdecay2_from_t_vec[i]),'frac_delta_pts':((Wdecay2_from_t_vec[i].pt - jet_vectors.pt)/Wdecay2_from_t_vec[i].pt)}
        elif parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][i] in had_range or parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][i] in had_range:
            particle_dict['Wdecay1_from_tbar'] = {'dRs':jet_vectors.deltaR(Wdecay1_from_tbar_vec[i]),'frac_delta_pts':((Wdecay1_from_tbar_vec[i].pt - jet_vectors.pt)/Wdecay1_from_tbar_vec[i].pt)}
            particle_dict['Wdecay2_from_tbar'] = {'dRs':jet_vectors.deltaR(Wdecay2_from_tbar_vec[i]),'frac_delta_pts':((Wdecay2_from_tbar_vec[i].pt - jet_vectors.pt)/Wdecay2_from_tbar_vec[i].pt)}
        else:
            print('WARNING: none of the W decays in event %d seem to be jets! What?!' % i)
            print('pdgids: %d, %d, %d, %d' % (parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][i] ))


        # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
        for par in particle_dict:

            # Get the previously calculated information
            dRs, pts = np.array(particle_dict[par]['dRs']), np.array(particle_dict[par]['frac_delta_pts'])

            # First select our pool of choices to those with dR <= dR_cut
            sel = np.array(dRs<=dR_cut)

            # If not allowing double matching, also remove jets that have already been matched
            if not allowDoubleMatching:
                sel = sel * np.array([True if j in jet_labels else False for j in range(n_jets)])

            # If this is a W decay product, we don't want to consider jets that are b-tagged (since fake rate is only ~1%)
            if par[0]=='W':
                sel = sel * np.array(jet_btags==0.)

            # Make these selections
            dRs_afterCuts = dRs[sel]
            pts_afterCuts = pts[sel]

            # If there is nothing left after the above selections, move on to the next particle
            if len(dRs_afterCuts)==0:
                continue

            # Else if there are also options with a decent fractional delta pt, prioritize these
            elif True in list(abs(pts_afterCuts)<=1):
                dRs_afterCuts = dRs_afterCuts[abs(pts_afterCuts)<=1]
                pts_afterCuts = pts_afterCuts[abs(pts_afterCuts)<=1]

            # Get the minimum dR from the cut down list, and find the jet and frac delta pt associated
            best_dR = np.min(dRs_afterCuts)
            best_jet = np.where(dRs==best_dR)[0][0]
            best_frac_delta_pt = pts[best_jet]

            # Sanity check:
            if par[0]=='W' and jet_btags[best_jet]==1.:
                print('WARNING: W matched to b jet')
            if best_dR > dR_cut:
                print('WARNING: saving best dR > dR_cut')

            # Save the best match we ended up with, and remove that jet from the list (so that it's not doubly assigned)
            event_matched_labels.append(best_jet)   # Save the matched jet label
            if not allowDoubleMatching: jet_labels.remove(best_jet) 

            # Also save the fractional delta pt between the particle and best jet
            useful_stuff.append([i,best_jet,par,best_dR,best_frac_delta_pt])
            

        # Get list of bools for whether or not jets are ttbar, and then append to array
        if len(event_matched_labels)==0:
            eventJetBools = [0 for j in range(n_jets)]
        else:
            eventJetBools = [1 if j in event_matched_labels else 0 for j in range(n_jets)]
        isttbarJet.append(eventJetBools)

        # Print every once in a while so we know there's progress
        if (i+1)%100000==0 or i==num_events-1:
            print ('Jet Events Processed: '+str(i+1))


    return isttbarJet, useful_stuff






# ---------- FUNCTION FIXING THE MNTUPLE ROOT FILES ---------- #

def fixRootFile(name,dR_cut,allowDoubleMatching=False):
    """
    Creates a new mntuple from the one given and fills in the empty histograms (W and b variables)
    Input: mntuple name (e.g.'1_parton_ejets')
    Output: saves new mntuple file as 'mntuple_ttbar_<name>_fixed.root'
    """

    # Just need this little string for file saving purposes
    dR_str = str(dR_cut).replace('.','d')

    # Open the original file and a new file to write to
    print('Opening root files ...')
    #og_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'.root')
    og_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed.root')
    fix_file = uproot.recreate('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed+match'+dR_str+'_new.root')

    # Get the reco and parton trees from the original file
    reco_tree = og_file['reco'].arrays()
    parton_tree = og_file['parton'].arrays()

    # Save the keys for later
    reco_keys = og_file['reco'].keys()
    parton_keys = og_file['parton'].keys()

    # Close the original file
    og_file.close()

    # Append W and b variables
    #parton_tree = appendWandB(parton_tree)

    # Remove events from the trees that we can't use
    print('Pruning trees ...')
    parton_tree, reco_tree = pruneTrees(parton_tree, reco_tree)

    # Get the jet matches
    print('Matching jets ...')
    isttbarJet, matching_info = getMatches(parton_tree, reco_tree, dR_cut, allowDoubleMatching)

    # Save the truth tags to the reco tree
    reco_tree['jet_isTruth'] = isttbarJet
    reco_keys.append('jet_isTruth')

    # Write the trees to the file
    print('Writing trees ...')
    fix_file['reco'] = {key:reco_tree[key] for key in reco_keys}
    fix_file['parton'] = {key:parton_tree[key] for key in parton_keys}
    print('Saved file: /data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed+match'+dR_str+'_new.root')

    # Close new file
    fix_file.close()

    # Save the matching info separately, since it's a weird shape
    np.save('/data/jchishol/mc16e/matching_info/matching_info_'+name+dR_str+'_new',matching_info)
    print('Saved file: /data/jchishol/mc16e/matching_info/matching_info_'+name+dR_str+'_new.npy')







# ---------- MAIN CODE ---------- #


# for i in range(8):
#     fixRootFile(str(i)+'_parton_mjets',dR_cut=0.6,allowDoubleMatching=True)

fixRootFile(str(5)+'_parton_mjets',dR_cut=0.6,allowDoubleMatching=True)
fixRootFile(str(6)+'_parton_mjets',dR_cut=0.6,allowDoubleMatching=True)
fixRootFile(str(7)+'_parton_mjets',dR_cut=0.6,allowDoubleMatching=True)











print('done :)')


