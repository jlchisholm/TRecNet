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
        jet_btags = reco_tree['jet_btagged'][i]

        # Create a set of jet labels (0 to jn-1) and matched labels (items will be moved from jet labels into matched labels as they're matched, if double matching is not allowed)
        jet_labels = list(range(n_jets))
        event_matched_labels = []

        # Get vectors of all the hadronic decay products (note: we want to start with the b quarks to ensure they get matched to btagged jets)
        particle_dict = {'b_from_t':b_from_t_vec[i],'b_from_tbar':b_from_tbar_vec[i]}
        if parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][i] in had_range or parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][i] in had_range:
            particle_dict['Wdecay1_from_t'] = Wdecay1_from_t_vec[i]
            particle_dict['Wdecay2_from_t'] = Wdecay2_from_t_vec[i]
        elif parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][i] in had_range or parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][i] in had_range:
            particle_dict['Wdecay1_from_tbar'] = Wdecay1_from_tbar_vec[i]
            particle_dict['Wdecay2_from_tbar'] = Wdecay2_from_tbar_vec[i]
        else:
            print('WARNING: none of the W decays in event %d seem to be jets! What?!' % i)
            print('pdgids: %d, %d, %d, %d' % (parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][i] ))

        # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
        for par in particle_dict:

            # Get particle vector
            particle_vec = particle_dict[par]

            # Create a variable to hold the best jet match and its dR and fractional delta pt (initializing to wild values to it will never be as large as)
            best_jet = -1
            best_dR = 1000.
            best_frac_delta_pt = 1000.

            # Create a list to hold the options that are within dR < dR_cut AND -1 < frac_delta_pt < 1
            good_options = []
                    
            # If we have a b-quark ...
            if par[0]=='b':

                # For each of the (unassigned) jets
                for j in jet_labels:

                    # Only want jets that are b-tagged when matching to b quark
                    if jet_btags[j]==1:

                        # Grab the jet vector
                        jet_vec = jet_vectors[j]

                        # Calculate dR between the current particle and jet combo, as well as the fractional delta pt
                        dR = particle_vec.deltaR(jet_vec)
                        frac_delta_pt = (particle_vec.pt - jet_vec.pt)/particle_vec.pt

                        # If in the target region, save this as a good option
                        if dR<=dR_cut and abs(frac_delta_pt)<=1:
                            good_options.append([j,dR,frac_delta_pt])

                        # If this dR is lower than the previously lowest, save this as the best dR
                        if dR < best_dR or (dR==best_dR and frac_delta_pt < best_frac_delta_pt): 
                            best_dR = dR
                            best_jet = j
                            best_frac_delta_pt = frac_delta_pt

                        
            # If we have a W decay product that is a jet ...
            else:

                # For each of the unassigned jets
                for j in jet_labels:

                    # Grab the jet vector
                    jet_vec = jet_vectors[j]

                    # Calculate dR, as well as the fractional delta pt
                    dR = particle_vec.deltaR(jet_vec)
                    frac_delta_pt = (particle_vec.pt - jet_vec.pt)/particle_vec.pt

                    # If in the target region, save this as a good option
                    if dR<=dR_cut and abs(frac_delta_pt)<=1:
                        good_options.append([j,dR,frac_delta_pt])
                    
                    # If this dR is lower than the previously lowest, save this as the best match
                    if dR < best_dR or (dR==best_dR and frac_delta_pt < best_frac_delta_pt): 
                        best_jet = j
                        best_dR = dR
                        best_frac_delta_pt = frac_delta_pt

            # If dR is too large, don't assign anything
            if best_dR <= dR_cut:

                # If we have some options in the desired region, prioritize these (if there are none, it will default to best dR)
                if good_options!=[]:

                    # Find the minimum dR within these options, get its index, and use it as the best match
                    min_dR = np.array(good_options)[:,1].min()
                    min_dR_index = np.where(np.array(good_options)[:,1]==min_dR)[0][0]
                    best_jet = good_options[min_dR_index][0]
                    best_dR = good_options[min_dR_index][1]
                    best_frac_delta_pt = good_options[min_dR_index][2]

                # Save the best match we ended up with, and remove that jet from the list (so that it's not doubly assigned)
                event_matched_labels.append(best_jet)   # Save the matched jet label
                if not allowDoubleMatching: jet_labels.remove(best_jet) 

                # Also save the fractional delta pt between the particle and best jet
                useful_stuff.append([i,best_jet,par,best_dR,best_frac_delta_pt])


        # Get list of bools for whether or not jets are ttbar, and then append to array
        if event_matched_labels == []:
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
    fix_file = uproot.recreate('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed+match'+dR_str+'.root')

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
    print('Pruning tress ...')
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
    print('Saved file: /data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed+match'+dR_str+'.root')

    # Close new file
    fix_file.close()

    # Save the matching info separately, since it's a weird shape
    np.save('matching_info/matching_info_'+name+dR_str,matching_info)
    print('Saved file: matching_info/matching_info_'+name+dR_str)







### Want to put all of this together at some point, but don't want to run it all at once for the time being


def addJetMatchLabels(name,dR_cut,allowDoubleMatching=False):

    print('Opening files ...')

    # Open the original file and a new file to write to
    og_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed.root')
    fix_file = uproot.recreate('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed+match06.root')

    # Get the reco and parton trees from the original file
    reco_tree = og_file['reco'].arrays()
    parton_tree = og_file['parton'].arrays()

    # Save the keys for later
    reco_keys = og_file['reco'].keys()
    parton_keys = og_file['parton'].keys()

    og_file.close()

    # Remove bad events from the trees
    print('Pruning trees ...')
    parton_tree, reco_tree = pruneTrees(parton_tree, reco_tree)



    print('Starting jet matching ...')    

    # Create a list to save all the matched labels in
    isttbarJet = []
    frac_delta_pt = []   # This list will not be separated by events or by jets, it's just one big long list

    # Array for counting how many jets are tagged but not amoung 6 highest pt jets 
    high_pt_counter = 0
    doub_match_counter = 0
    no_match_counter = 0

    # Need to go through event by event :(
    num_events = len(reco_tree['eventNumber'])
    for i in range(num_events):

        # Get the number of jets in this event
        n_jets = int(reco_tree['jet_n'][i])

        # Calculate the jet vectors for this event, as well as get the btags
        jet_vectors = vector.arr({"pt":reco_tree['jet_pt'][i],"eta":reco_tree['jet_eta'][i],"phi":reco_tree['jet_phi'][i],"E":reco_tree['jet_e'][i]})
        jet_btags = reco_tree['jet_btagged'][i]

        # Create a set of jet labels (0 to jn-1) and matched labels (items will be moved from jet labels into matched labels as they're matched)
        jet_labels = list(range(n_jets))
        event_matched_labels = []



        # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
        particle_list = ['b_from_t','b_from_tbar']
        if parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][i] in had_range or parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][i] in had_range:
            particle_list.append('Wdecay1_from_t')
            particle_list.append('Wdecay2_from_t') 
        elif parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][i] in had_range or parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][i] in had_range:
            particle_list.append('Wdecay1_from_tbar')
            particle_list.append('Wdecay2_from_tbar')
        else:
            print('WARNING: none of the W decays in event %d seem to be jets! What?!' % i)
            print('pdgids: %d, %d, %d, %d' % (parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][i] ))



        # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
        for par in particle_list:

            # Calculate the vector for the particle of interest
            # ('Wdecay2_from_t_eta' is full of zeroes, so we need to approximate with y for now)
            if par=='Wdecay2_from_t':
                particle_vec = vector.obj(pt=parton_tree['MC_'+par+'_afterFSR_pt'][i],eta=parton_tree['MC_'+par+'_afterFSR_y'][i],phi=parton_tree['MC_'+par+'_afterFSR_phi'][i],E=parton_tree['MC_'+par+'_afterFSR_E'][i])
            else:
                particle_vec = vector.obj(pt=parton_tree['MC_'+par+'_afterFSR_pt'][i],eta=parton_tree['MC_'+par+'_afterFSR_eta'][i],phi=parton_tree['MC_'+par+'_afterFSR_phi'][i],E=parton_tree['MC_'+par+'_afterFSR_E'][i])

            # Create a variable to hold the best jet match and dR calculated
            best_jet = -1
            best_dR = 1000.
                    
            # If we have a b-quark ...
            if par.split('_')[0]=='b':

                # For each of the unassigned jets
                for j in jet_labels:

                    # Only want jets that are b-tagged when matching to b quark
                    if jet_btags[j]==1:

                        # Calculate dR between the current particle and jet combo
                        dR = particle_vec.deltaR(jet_vectors[j])

                        # If this dR is lower than the previously lowest, save this as the best dR
                        if dR < best_dR: 
                            best_dR = dR
                            best_jet = j
                        
            # If we have a W decay product that is a jet ...
            else:

                # For each of the unassigned jets
                for j in jet_labels:

                    # Calculate dR 
                    dR = particle_vec.deltaR(jet_vectors[j])
                    
                    # If this dR is lower than the previously lowest, save this as the best dR
                    if dR < best_dR: 
                        best_dR = dR
                        best_jet = j

            # If dR is too large, don't assign anything
            if best_dR > dR_cut:
                no_match_counter+=1
            else:
                # If we had this jet in the list of matched events already
                if event_matched_labels!=[] and best_jet in event_matched_labels: doub_match_counter+=1

                # Save the best match we ended up with, and remove that jet from the list (so that it's not doubly assigned)
                event_matched_labels.append(best_jet)   # Save the matched jet label
                if not allowDoubleMatching: jet_labels.remove(best_jet) 

                # Also save the fractional delta pt between the particle and best jet
                fdpt = (parton_tree['MC_'+par+'_afterFSR_pt'][i] - reco_tree['jet_pt'][i][best_jet])/parton_tree['MC_'+par+'_afterFSR_pt'][i]
                frac_delta_pt.append([i,best_jet,par,best_dR,fdpt])

                # If this was not within the 6 highest pt jets, add to the counter
                if best_jet >=6: high_pt_counter+=1

            

        # Get bools for whether or not jets are ttbar, and then save to array
        if event_matched_labels == []:
            eventJetBools = [0 for j in range(n_jets)]
        else:
            eventJetBools = [1 if j in event_matched_labels else 0 for j in range(n_jets)]
        isttbarJet.append(eventJetBools)

        # Sanity check
        #if len(np.array(event_matched_labels)[:,0])!=4:
        #    print('WARNING: more or less than 4 particles were matched to the jets! Ah!')

        # Print every once in a while so we know there's progress
        if (i+1)%100000==0 or i==num_events-1:
            print ('Jet Events Processed: '+str(i+1))

    
    # Save the truth tags to the reco tree
    reco_tree['jet_isTruth'] = isttbarJet
    reco_keys.append('jet_isTruth')

    # Save the fractional delta pt calculations to the reco tree
    #reco_tree['jet_matchFracDelta_pt'] = np.array(frac_delta_pt)

    print("Number of jets matched but not within 6 highest pt jets: ", high_pt_counter)
    print("Total number of matched jets (i.e. 4*total number of events): ", 4*len(reco_tree['eventNumber']))
    print("Fraction of these jets: ", high_pt_counter/(4*len(reco_tree['eventNumber'])))

    print("Number of jets matched to two products: ", doub_match_counter)
    print("Total number of matched jets (i.e. 4*total number of events): ", 4*len(reco_tree['eventNumber']))
    print("Fraction of these jets: ", doub_match_counter/(4*len(reco_tree['eventNumber'])))

    print('Writing trees ...')

    # Write the trees to the file

    fix_file['reco'] = {key:reco_tree[key] for key in reco_keys}
    fix_file['parton'] = {key:parton_tree[key] for key in parton_keys}

    fix_file.close()

    # Save the fractional delta pt separately, since it's a weird shape
    np.save('frac_delta_pt/frac_delta_pt_'+name+'_06',frac_delta_pt)









### Want to put all of this together at some point, but don't want to run it all at once for the time being


def addJetMatchLabels2(name,dR_cut,allowDoubleMatching=False):

    dR_str = str(dR_cut).replace('.','d')

    print('Opening files ...')

    # Open the original file and a new file to write to
    og_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed.root')
    fix_file = uproot.recreate('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed+match'+dR_str+'.root')

    # Get the reco and parton trees from the original file
    reco_tree = og_file['reco'].arrays()
    parton_tree = og_file['parton'].arrays()

    # Save the keys for later
    reco_keys = og_file['reco'].keys()
    parton_keys = og_file['parton'].keys()

    og_file.close()

    # Remove events from the trees that we can't use
    print('Pruning trees ...')
    parton_tree, reco_tree = pruneTrees(parton_tree, reco_tree)


    print('Starting jet matching ...')    

    # Create a list to save all the matched labels in
    isttbarJet = []
    useful_stuff = []   # This list will not be separated by events or by jets, it's just one big long list

    # Array for counting how many jets are tagged but not amoung 6 highest pt jets 
    high_pt_counter = 0
    doub_match_counter = 0
    no_match_counter = 0

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
        jet_btags = reco_tree['jet_btagged'][i]

        # Create a set of jet labels (0 to jn-1) and matched labels (items will be moved from jet labels into matched labels as they're matched)
        jet_labels = list(range(n_jets))
        event_matched_labels = []



        # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
        particle_dict = {'b_from_t':b_from_t_vec[i],'b_from_tbar':b_from_tbar_vec[i]}
        if parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][i] in had_range or parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][i] in had_range:
            particle_dict['Wdecay1_from_t'] = Wdecay1_from_t_vec[i]
            particle_dict['Wdecay2_from_t'] = Wdecay2_from_t_vec[i]
        elif parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][i] in had_range or parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][i] in had_range:
            particle_dict['Wdecay1_from_tbar'] = Wdecay1_from_tbar_vec[i]
            particle_dict['Wdecay2_from_tbar'] = Wdecay2_from_tbar_vec[i]
        else:
            print('WARNING: none of the W decays in event %d seem to be jets! What?!' % i)
            print('pdgids: %d, %d, %d, %d' % (parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][i] ,parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][i] ))



        # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
        for par in particle_dict:

            # Get particle vector
            particle_vec = particle_dict[par]

            # Create a variable to hold the best jet match and dR calculated
            best_jet = -1
            best_dR = 1000.
            best_frac_delta_pt = 1000.


            # Create a list to hold the options that are within dR < dR_cut AND -1 < frac_delta_pt < 1
            good_options = []
                    
            # If we have a b-quark ...
            if par.split('_')[0]=='b':

                # For each of the unassigned jets
                for j in jet_labels:

                    # Only want jets that are b-tagged when matching to b quark
                    if jet_btags[j]==1:

                        # Grab the jet vector
                        jet_vec = jet_vectors[j]

                        # Calculate dR between the current particle and jet combo, as well as the fractional delta pt
                        dR = particle_vec.deltaR(jet_vec)
                        frac_delta_pt = (particle_vec.pt - jet_vec.pt)/particle_vec.pt

                        # If in the target region, save this as a good option
                        if dR<=dR_cut and abs(frac_delta_pt)<=1:
                            good_options.append([j,dR,frac_delta_pt])

                        # If this dR is lower than the previously lowest, save this as the best dR
                        if dR < best_dR: 
                            best_dR = dR
                            best_jet = j
                            best_frac_delta_pt = frac_delta_pt
                        
            # If we have a W decay product that is a jet ...
            else:

                # For each of the unassigned jets
                for j in jet_labels:

                    # Grab the jet vector
                    jet_vec = jet_vectors[j]

                    # Calculate dR, as well as the fractional delta pt
                    dR = particle_vec.deltaR(jet_vec)
                    frac_delta_pt = (particle_vec.pt - jet_vec.pt)/particle_vec.pt

                    # If in the target region, save this as a good option
                    if dR<=dR_cut and abs(frac_delta_pt)<=1:
                        good_options.append([j,dR,frac_delta_pt])
                    
                    # If this dR is lower than the previously lowest, save this as the best match
                    if dR < best_dR: 
                        best_jet = j
                        best_dR = dR
                        best_frac_delta_pt = frac_delta_pt

            # If dR is too large, don't assign anything
            if best_dR > dR_cut:
                no_match_counter+=1
            else:
                # If we have some options in the desired region, prioritize these (if there are none, it will default to best dR)
                if good_options!=[]:

                    # Find the minimum dR within these options, get its index, and use it as the best match
                    min_dR = np.array(good_options)[:,1].min()
                    min_dR_index = np.where(np.array(good_options)[:,1]==min_dR)[0][0]
                    best_jet = good_options[min_dR_index][0]
                    best_dR = good_options[min_dR_index][1]
                    best_frac_delta_pt = good_options[min_dR_index][2]

                # If we had this jet in the list of matched events already
                if event_matched_labels!=[] and best_jet in event_matched_labels: doub_match_counter+=1

                # Save the best match we ended up with, and remove that jet from the list (so that it's not doubly assigned)
                event_matched_labels.append(best_jet)   # Save the matched jet label
                if not allowDoubleMatching: jet_labels.remove(best_jet) 

                # Also save the fractional delta pt between the particle and best jet
                useful_stuff.append([i,best_jet,par,best_dR,best_frac_delta_pt])

                # If this was not within the 6 highest pt jets, add to the counter
                if best_jet >=6: high_pt_counter+=1

            

        # Get bools for whether or not jets are ttbar, and then save to array
        if event_matched_labels == []:
            eventJetBools = [0 for j in range(n_jets)]
        else:
            eventJetBools = [1 if j in event_matched_labels else 0 for j in range(n_jets)]
        isttbarJet.append(eventJetBools)

        # Sanity check
        #if len(np.array(event_matched_labels)[:,0])!=4:
        #    print('WARNING: more or less than 4 particles were matched to the jets! Ah!')

        # Print every once in a while so we know there's progress
        if (i+1)%100000==0 or i==num_events-1:
            print ('Jet Events Processed: '+str(i+1))

    
    # Save the truth tags to the reco tree
    reco_tree['jet_isTruth'] = isttbarJet
    reco_keys.append('jet_isTruth')

    # Save the fractional delta pt calculations to the reco tree
    #reco_tree['jet_matchFracDelta_pt'] = np.array(frac_delta_pt)

    print("Number of jets matched but not within 6 highest pt jets: ", high_pt_counter)
    print("Total number of matched jets (i.e. 4*total number of events): ", 4*len(reco_tree['eventNumber']))
    print("Fraction of these jets: ", high_pt_counter/(4*len(reco_tree['eventNumber'])))

    print('-------------------')

    print("Number of jets matched to two products: ", doub_match_counter)
    print("Total number of matched jets (i.e. 4*total number of events): ", 4*len(reco_tree['eventNumber']))
    print("Fraction of these jets: ", doub_match_counter/(4*len(reco_tree['eventNumber'])))

    print('Writing trees ...')

    # Write the trees to the file

    fix_file['reco'] = {key:reco_tree[key] for key in reco_keys}
    fix_file['parton'] = {key:parton_tree[key] for key in parton_keys}

    fix_file.close()

    # Save the fractional delta pt separately, since it's a weird shape
    np.save('matching_info/matching_info_'+name+dR_str,useful_stuff)









    




# ---------- MAIN CODE ---------- #

#fixRootFile('5_parton_mjets')

fixRootFile('2_parton_ejets',dR_cut=0.4,allowDoubleMatching=True)
fixRootFile('2_parton_ejets',dR_cut=0.6,allowDoubleMatching=True)








print('done :)')


