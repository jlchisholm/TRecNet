import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import vector


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

    # Get the reco and parton trees
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




# ---------- MAIN CODE ---------- #

fixRootFile('5_parton_mjets')
fixRootFile('6_parton_mjets')
fixRootFile('7_parton_mjets')

print('done :)')


