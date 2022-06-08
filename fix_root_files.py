import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('GTK3Agg')   # need this one if you want plots displayed
import vector


# Define some helpful ranges
had_range = list(range(1,9))              # Quarks have pdgid 1 to 8
lep_range = list(range(11,19))            # Leptons have pdgid 11 to 18
bos_range = list(range(21,26))+[9,37]     # Bosons have pdgid 21 to 25, and 9, and 37


# ---------- FUNCTION FIXING THE MNTUPLE ROOT FILES ---------- #

def fixRootFile(name):
    """
    Creates a new mntuple from the one given and fills in the empty histograms (W and b variables)
    Input: mntuple name (e.g.'1_parton_ejets')
    Output: saves new mntuple file as 'mntuple_ttbar_<name>_fixed.root'
    """

    print('Opening root files ...')

    # Open the original file and a new file to write to
    og_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'.root')
    fix_file = uproot.recreate('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed.root')

    # Get the reco and parton trees from the original file
    reco_tree = og_file['reco'].arrays()
    parton_tree = og_file['parton'].arrays()

    print('Writing reco tree ...')

    # Write the reco tree to the file (there were no issues with this one)
    fix_file['reco'] = {key:reco_tree[key] for key in og_file['reco'].keys()}

    # For both the t and tbar sides ...
    for t in ['t','tbar']:

        print('Calculating 4-vectors for '+t+' ...')

        # Create lists for the variables
        W_variables = []
        b_variables = []

        # Run through each event
        num_events = len(parton_tree['isDummy'])
        for i in range(num_events):

            # Get W decay vectors
            # ('Wdecay2_from_t_eta' is full of zeroes, so we need to approximate with y for now)
            if t=='t':
                Wdecay2_vec = vector.obj(pt=parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_pt'][i],phi=parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_phi'][i],eta=parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_y'][i],mass=parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_m'][i])
            else:
                Wdecay2_vec = vector.obj(pt=parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_pt'][i],phi=parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_phi'][i],eta=parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_eta'][i],mass=parton_tree['MC_Wdecay2_from_'+t+'_afterFSR_m'][i])
            Wdecay1_vec = vector.obj(pt=parton_tree['MC_Wdecay1_from_'+t+'_afterFSR_pt'][i],phi=parton_tree['MC_Wdecay1_from_'+t+'_afterFSR_phi'][i],eta=parton_tree['MC_Wdecay1_from_'+t+'_afterFSR_eta'][i],mass=parton_tree['MC_Wdecay1_from_'+t+'_afterFSR_m'][i])

            # Add decays together to get W vector
            W_vec = Wdecay1_vec + Wdecay2_vec

            # Get t or tbar vector
            t_vec = vector.obj(pt=parton_tree['MC_'+t+'_afterFSR_pt'][i],phi=parton_tree['MC_'+t+'_afterFSR_phi'][i],eta=parton_tree['MC_'+t+'_afterFSR_eta'][i],mass=parton_tree['MC_'+t+'_afterFSR_m'][i])

            # Subtract W from t to get b quark vector
            b_vec = t_vec - W_vec

            # Append vector values to list
            W_variables.append([W_vec.pt,W_vec.eta,W_vec.rapidity,W_vec.phi,W_vec.mass,W_vec.E])
            b_variables.append([b_vec.pt,b_vec.eta,b_vec.rapidity,b_vec.phi,b_vec.mass,b_vec.E])

        print('Appending values to tree ...')

        # Put values in tree
        for i, var in enumerate(['pt','eta','y','phi','m','E']):
            parton_tree['MC_W_from_'+t+'_afterFSR_'+var] = np.array(W_variables)[:,i]
            parton_tree['MC_b_from_'+t+'_afterFSR_'+var] = np.array(b_variables)[:,i]

    print('Writing parton tree ...')

    # Write the fixed parton tree to the file
    fix_file['parton'] = {key:parton_tree[key] for key in og_file['parton'].keys()}

    fix_file.close()
    og_file.close()



### Want to put all of this together at some point, but don't want to run it all at once for the time being


def addJetMatchLabels(name):

    print('Opening files ...')

    # Open the original file and a new file to write to
    og_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed.root')
    fix_file = uproot.recreate('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed+match.root')

    # Get the reco and parton trees from the original file
    reco_tree = og_file['reco'].arrays()
    parton_tree = og_file['parton'].arrays()


    print('Pruning trees ...')

    # Not sure if I'll keep this here, but make necessary cuts on events we can't really use:
    # Remove events with nan y values
    for var in ['MC_thad_afterFSR_y','MC_tlep_afterFSR_y','MC_Wdecay1_from_t_afterFSR_y','MC_Wdecay2_from_t_afterFSR_y','MC_Wdecay1_from_tbar_afterFSR_y','MC_Wdecay2_from_tbar_afterFSR_y']:
        sel = np.invert(np.isnan(parton_tree[var]))
        parton_tree = parton_tree[sel]
        reco_tree = reco_tree[sel]

    # Remove events with met_met < 20 GeV
    sel = reco_tree['met_met']/1000 >= 20
    parton_tree = parton_tree[sel]
    reco_tree = reco_tree[sel]

    # Remove isDummy events and !isMatched events
    sel = reco_tree['isMatched']==1
    parton_tree = parton_tree[sel]
    reco_tree = reco_tree[sel]
    sel = parton_tree['isDummy']==0
    parton_tree = parton_tree[sel]
    reco_tree = reco_tree[sel]











    print('Starting jet matching ...')    

    # Create a list to save all the matched labels in
    isttbarJet = []
    frac_delta_pt = []   # This list will not be separated by events or by jets, it's just one big long list

    # Array for counting how many jets are tagged but not amoung 6 highest pt jets 
    counter = 0

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

            # Save the best match we ended up with, and remove that jet from the list (so that it's not doubly assigned)
            frac_delta_pt.append((parton_tree['MC_'+par+'_afterFSR_pt'][i] - reco_tree['jet_pt'][i][j])/parton_tree['MC_'+par+'_afterFSR_pt'][i])
            event_matched_labels.append([best_jet,par,best_dR])   # Save the matched jet label, the particle label, and the dR
            jet_labels.remove(best_jet) 

            # If this was not within the 6 highest pt jets, add to the counter
            if best_jet >=6: counter+=1


        # Get bools for whether or not jets are ttbar, and then save to array
        jetBools = [1 if j in np.array(event_matched_labels)[:,0].astype(int) else 0 for j in range(n_jets)]
        isttbarJet.append(jetBools)

        # Sanity check
        if len(np.array(event_matched_labels)[:,0])!=4:
            print('WARNING: more or less than 4 particles were matched to the jets! Ah!')

        # Print every once in a while so we know there's progress
        if (i+1)%50000==0 or i==num_events-1:
            print ('Jet Events Processed: '+str(i+1))

    
    # Save the truth tags to the reco tree
    reco_tree['jet_isTruth'] = isttbarJet

    # Save the fractional delta pt calculations to the reco tree
    #reco_tree['jet_matchFracDelta_pt'] = np.array(frac_delta_pt)

    print("Number of jets matched but not within 6 highest pt jets: ", counter)
    print("Total number of matched jets (i.e. 4*total number of events): ", 4*len(reco_tree['eventNumber']))
    print("Fraction of these jets: ", counter/(4*len(reco_tree['eventNumber'])))

    print('Writing trees ...')

    # Write the trees to the file
    fix_file['reco'] = {key:reco_tree[key] for key in og_file['reco'].keys()}
    fix_file['parton'] = {key:parton_tree[key] for key in og_file['parton'].keys()}

    fix_file.close()
    og_file.close()

    # Save the fractional delta pt separately, since it's a weird shape
    np.save('frac_delta_pt_'+name,frac_delta_pt)


    




# ---------- MAIN CODE ---------- #

#fixRootFile('5_parton_mjets')
#frac_delta_pt = addJetMatchLabels('0_parton_ejets')

# Save a histogram of the fractional delta pt
data = np.load('frac_delta_pt_0_parton_ejets.npy')
print(max(data))
print(min(data))
print(data[data<0.9])
good_data = data[data<0.9]
good_data = good_data[good_data>-0.9]
print(len(good_data)/len(data))

plt.figure('frac_delta_pt')
plt.hist(data,bins=30,range=(-1,1),histtype='step')
plt.xlabel('frac_delta_pt')
plt.ylabel('Counts')
plt.show()




print('done :)')


