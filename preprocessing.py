import uproot
import h5py
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import vector
from scipy.optimize import linear_sum_assignment

# Define some helpful ranges
had_range = list(range(1,9))              # Quarks have pdgid 1 to 8
lep_range = list(range(11,19))            # Leptons have pdgid 11 to 18
bos_range = list(range(21,26))+[9,37]     # Bosons have pdgid 21 to 25, and 9, and 37


# ---------- FUNCTION FOR MATCHING JETS AND DECAY PRODUCTS ---------- #

def getJetMatching(event,parton_tree,jet_vectors,btags,num_jets):
    """
    Matches the four decay products to four out of however many jets in a given event
    Input: event number, parton level tree, vectors (vector.arr) for the jets in the event, btags for the jets in the event, number of jets in the event
    Output: list of [jet label (0 to jn-1), name of the matching decay product (e.g.'b_from_tbar')]
    """
    
    #print("Matching event:",event)
    #print("Number of jets: ",num_jets)
    #print('---------------------------------------\n')

    # Create a set of jet labels (0 to jn-1) and matched labels (items will be moved from jet labels into matched labels as they're matched)
    jet_labels = list(range(num_jets))
    matched_labels = []

    # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
    particle_list = ['b_from_t','b_from_tbar']
    if parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][event] in had_range or parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][event] in had_range:
        particle_list.append('Wdecay1_from_t')
        particle_list.append('Wdecay2_from_t') 
    elif parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][event] in had_range or parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][event] in had_range:
        particle_list.append('Wdecay1_from_tbar')
        particle_list.append('Wdecay2_from_tbar')
    else:
        print('WARNING: none of the W decays seem to be jets! What?!')

    # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
    for par in particle_list:

        #print('Looking at decay product '+par)

        # Calculate the vector for the particle of interest
        particle_vec = vector.obj(pt=parton_tree['MC_'+par+'_afterFSR_pt'][event],eta=parton_tree['MC_'+par+'_afterFSR_eta'][event],phi=parton_tree['MC_'+par+'_afterFSR_phi'][event],E=parton_tree['MC_'+par+'_afterFSR_E'][event])
        #print("particle_vec is giving:",particle_vec)

        # Create a list to hold the dR values
        dRs = []
        labels = []
                
        # If we have a b-quark ...
        if par.split('_')[0]=='b':

            # For each of the unassigned jets
            for label in jet_labels:

                #print("On jet #",label)

                # Only want jets that are b-tagged when matching to b quark
                if btags[label]==1:

                    # Grab the jet vector
                    jet_vec = jet_vectors[label]
                    #print("jet_vec is giving:",jet_vec)

                    # Calculate dR and append it to a list
                    #print("dR=",particle_vec.deltaR(jet_vec))
                    dRs.append(particle_vec.deltaR(jet_vec))
                    labels.append(label)
                    
        # If we have a W decay product that is a jet ...
        else:

            # For each of the unassigned jets
            for label in jet_labels:

                #print("On jet #",label)

                # Grab the jet vector
                jet_vec = jet_vectors[label]
                #print("jet_vec is giving:",jet_vec)

                # Calculate dR and append it to a list
                #print("dR=",particle_vec.deltaR(jet_vec))
                dRs.append(particle_vec.deltaR(jet_vec))
                labels.append(label)


        # Figuring out which jet matches the decay product best!
        #print("dRs: ",dRs)
        min_index = np.where(dRs==min(dRs))[0][0]  # Get the index where the minimum dR occurs
        j_label = labels[min_index]            # Get the jet label corresponding to the minimum dR
        matched_labels.append([j_label,par])   # Put the matched jet label in the list, as well as the particle label so we know which jet matched which particle
        jet_labels.remove(j_label)             # Remove that number from the list of jet labels now that it's been used

        #print('---------------------------------------\n')

    
    #print(matched_labels)
    # Little check that things went okay
    if len(matched_labels)!=4:
        print('WARNING: more or less than 4 particles were matched to the jets! Ah!')


    return matched_labels




# ---------- FUNCTION FOR MATCHING JETS AND DECAY PRODUCTS ---------- #


## This one attempts to use the Hungarian algorithm
def getJetMatching_HungarianMethod(event,parton_tree,jet_vectors,btags,num_jets):
    """
    Matches the four decay products to four out of however many jets in a given event
    Input: event number, parton level tree, vectors (vector.arr) for the jets in the event, btags for the jets in the event, number of jets in the event
    Output: list of [jet label (0 to jn-1), name of the matching decay product (e.g.'b_from_tbar')]
    """

    #print("Matching event:",event)
    #print("Number of jets: ",num_jets)
    #print('---------------------------------------\n')

    # Run through b quarks first, then (hadronic) W decay products, matching the objects to the closest reconstructed jet (by dR)
    particle_list = ['b_from_t','b_from_tbar']
    if parton_tree['MC_Wdecay1_from_t_afterFSR_pdgid'][event] in had_range or parton_tree['MC_Wdecay2_from_t_afterFSR_pdgid'][event] in had_range:
        particle_list.append('Wdecay1_from_t')
        particle_list.append('Wdecay2_from_t') 
    elif parton_tree['MC_Wdecay1_from_tbar_afterFSR_pdgid'][event] in had_range or parton_tree['MC_Wdecay2_from_tbar_afterFSR_pdgid'][event] in had_range:
        particle_list.append('Wdecay1_from_tbar')
        particle_list.append('Wdecay2_from_tbar')
    else:
        print('WARNING: none of the W decays seem to be jets! What?!')

    # Sanity check:
    if len(particle_list)!=4:
        print('WARNING: more or less than 4 decay products are being matched to the jets! Ah!')


    # Go through and calculate a cost matrix (based on dR and btags)
    cost_matrix = []
    for par in particle_list:

        #print('Looking at decay product '+par)

        # Calculate the vector for the particle of interest
        particle_vec = vector.obj(pt=parton_tree['MC_'+par+'_afterFSR_pt'][event],eta=parton_tree['MC_'+par+'_afterFSR_eta'][event],phi=parton_tree['MC_'+par+'_afterFSR_phi'][event],E=parton_tree['MC_'+par+'_afterFSR_E'][event])
        #print("particle_vec is giving:",particle_vec)


        # Create a list to hold the dR values
        dR_list = []

        # For each of the jets
        for j in range(num_jets):

            #print("On jet #",j)

            # Grab the jet vector
            jet_vec = jet_vectors[j]
            #print("jet_vec is giving:",jet_vec)

            # Calculate dR
            dR = particle_vec.deltaR(jet_vec)

            # Sanity check:
            if dR>20:
                print('WARNING: dR is a little big check it out')

            # If we are looking at a b quark and the jet isn't b-tagged, add a huge cost to it
            if par.split('_')[0]=='b' and btags[j]!=1:
                dR+= 10000

            # Append dR to the list
            dR_list.append(dR)

        #print("dR:",dR_list)

        # Append dR list for the jets with this particle to the cost matrix
        cost_matrix.append(dR_list)
    
        #print('------------------\n')

    # Use the Hungarian method to get the best matching
    cost_matrix = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind,col_ind].sum()

    #print("Cost matrix:",cost_matrix)
    #print("rows:",row_ind)
    #print("cols:",col_ind)
    #print("cost:",cost)

    # Get the matching labels for the decay products and jets
    matched_pairs = []
    for i in range(len(row_ind)):
        pmatch = particle_list[row_ind[i]]  # Get decay product (string) that was matched
        jmatch = col_ind[i]                 # Get jet (int) that was matched to it
        matched_pairs.append([jmatch,pmatch])

    #print(matched_pairs)


    return matched_pairs, cost







# ---------- FUNCTION FOR OBTAINING JET DATA ---------- #

def getJetsData(parton_tree,reco_tree,withJetMatch,jn=6):
    """
    Creates an array of dataframes for the jets
    Input: parton level tree from ROOT file, reco level tree from ROOT file, bool for if you want to match jets, number of jets <jn> want to have per event (default is 6)
    Output: an array of <jn> dataframes (one for each of the jets)
    """

    # Get the total number of events
    num_events = len(reco_tree['eventNumber'])
    print('Number of events: '+str(num_events))

    # Create temporary jet arrays
    jets = [[] for _ in range(jn)]

    for n in range(num_events):     # Going through each event

        # Get the variables for the jets in this event
        pts = reco_tree['jet_pt'][n]
        etas = reco_tree['jet_eta'][n]
        phis = reco_tree['jet_phi'][n]
        es = reco_tree['jet_e'][n]
        btags = reco_tree['jet_btagged'][n]

        # Get the number of jets in this event
        n_jets = int(reco_tree['jet_n'][n])

        # Calculate the jet vectors
        jet_vectors = vector.arr({"pt":pts,"eta":etas,"phi":phis,"E":es})

        # Calculate mass for the jet vectors
        ms = jet_vectors.mass

        
        if withJetMatch:

            # Jet and decay product matching
            #matched_pairs = getJetMatching(n,parton_tree,jet_vectors,btags,n_jets)
            matched_pairs, cost = getJetMatching_HungarianMethod(n,parton_tree,jet_vectors,btags,n_jets)

            # Append data for the jets
            for i in range(jn):
                if i+1<=n_jets:  # Append like normal for however many jets are in the event
                    isTrue = 1 if i in np.array(matched_pairs)[:,0].astype(int) else 0
                    jets[i].append([pts[i],etas[i],phis[i],ms[i],btags[i],isTrue])
                else:          # Otherwise pad the jets with constants (with 0's for isTrue, since it's not even a real jet)
                    jets[i].append([0.,0.,0.,0.,-1.,0.]) 

        else:

            # Append data for the jets
            for i in range(jn):
                if i+1<=n_jets:  # Append like normal for however many jets are in the event
                    jets[i].append([pts[i],etas[i],phis[i],ms[i],btags[i]])
                else:          # Otherwise pad the jets with constants
                    jets[i].append([0.,0.,0.,0.,-1.]) 


        # Print every once in a while so we know there's progress
        if (n+1)%100000==0 or n==num_events-1:
            print ('Jet Events Processed: '+str(n+1))
            if withJetMatch: print('Sample of matched labels: ',matched_pairs)
            if withJetMatch: print('Sample of cost: ', cost)

    # Create an array of jet dataframes
    df_jets = [[] for _ in range(jn)]
    for i in range(jn):

        if withJetMatch:
            column_names = ('pt','eta','phi','m','isbtag','isTrue')
        else:
            column_names = ('pt','eta','phi','m','isbtag')

        df_jets[i] = pd.DataFrame(jets[i],columns=column_names)

        # Convert from MeV to GeV
        for v in ['pt','m']:
            df_jets[i][v] = df_jets[i][v]/1000

        # Print a progress statement
        print ('Jet '+str(i+1)+' dataframe created.')

    return df_jets


# ---------- FUNCTION FOR OBTAINING OTHER DATA ---------- #

def getOtherData(reco_tree):
    """
    Create a dataframe for the other (lep, met) variables
    Input: reco level tree from ROOT file 
    Output: dataframe for the other (lep, met) variables
    """

    # Create dataframe for other variables
    df_other = ak.to_pandas(reco_tree['isMatched'])
    df_other.rename(columns={'values':'isMatched'},inplace=True)
    df_other['lep_pt'] = reco_tree['lep_pt']/1000     # Convert from MeV to GeV
    df_other['lep_eta'] = reco_tree['lep_eta']
    df_other['lep_phi'] = reco_tree['lep_phi']
    df_other['met_met'] = reco_tree['met_met']/1000   # Convert from MeV to GeV
    df_other['met_phi'] = reco_tree['met_phi']

    print('Other dataframe created.')

    return df_other


# ---------- FUNCTION FOR OBTAINING TRUTH DATA ---------- #

def getTruthData(parton_tree,ttbar_addon):
    """
    Create a dataframe for the truth variables
    Input: parton level tree from ROOT file, true or false for added ttbar variables
    Output: dataframe for the truth variables
    """

    # Create temporary df for W truth variables (to help sort the hadronic and leptonic)
    df_truth_W = ak.to_pandas(parton_tree['isDummy'])
    df_truth_W.rename(columns={'values':'isDummy'},inplace=True)

    # Append the pdgid values for each of the four W decay products
    for par in ['Wdecay1_from_tbar_','Wdecay1_from_t_','Wdecay2_from_tbar_','Wdecay2_from_t_']:
        df_truth_W[par+'pdgid'] = parton_tree['MC_'+par+'afterFSR_pdgid']

    # Append the variables for W's
    for par in ['MC_W_from_t_afterFSR_','MC_W_from_tbar_afterFSR_']:
        for var in ['pt','eta','y','phi','m','E']:
            df_truth_W[par+var] = parton_tree[par+var]


    # Determine if the W decay from tbar is a hadronic decay
    df_truth_W['Wdecay_from_tbar_isHadronicDecay'] = df_truth_W.apply(lambda x : True if abs(x['Wdecay1_from_tbar_pdgid']) in had_range and abs(x['Wdecay2_from_tbar_pdgid']) in had_range else False if abs(x['Wdecay1_from_tbar_pdgid']) in lep_range and abs(x['Wdecay2_from_tbar_pdgid']) in lep_range  else 'Error: Boson' if abs(x['Wdecay1_from_tbar_pdgid']) in bos_range or abs(x['Wdecay2_from_tbar_pdgid']) in bos_range else 'Error: 0' if abs(x['Wdecay1_from_tbar_pdgid'])==0 or abs(x['Wdecay2_from_tbar_pdgid'])==0 else 'Error: Mismatch' ,axis=1)

    # Determine if the W decay from t is a hadronic decay
    df_truth_W['Wdecay_from_t_isHadronicDecay'] = df_truth_W.apply(lambda x : True if abs(x['Wdecay1_from_t_pdgid']) in had_range and abs(x['Wdecay2_from_t_pdgid']) in had_range else False if abs(x['Wdecay1_from_t_pdgid']) in lep_range and abs(x['Wdecay2_from_t_pdgid']) in lep_range  else 'Error: Boson' if abs(x['Wdecay1_from_t_pdgid']) in bos_range or abs(x['Wdecay2_from_t_pdgid']) in bos_range else 'Error: 0' if abs(x['Wdecay1_from_t_pdgid'])==0 or abs(x['Wdecay2_from_t_pdgid'])==0 else 'Error: Mismatch' ,axis=1)

    # Check that we don't have any funny business going on
    if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: Boson').any():      # Don't want any bosons from tbar
        print("WARNING: Decay 1 and/or 2 from tbar identified as boson.")
    if (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: Boson').any():         # Don't want any bosons from t
        print("WARNING: Decay 1 and/or 2 from t identified as boson.")
    if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: 0').any():          # Don't want any 0 pdgid from tbar
        print("WARNING: Decay 1 and/or 2 from tbar has 0 pdgid.")
    if (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: 0').any():             # Don't want any 0 pdgid from t
        print("WARNING: Decay 1 and/or 2 from t has 0 pdgid.")
    if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: Mismatch').any():   # Want decay 1 and 2 from tbar to agree
        print("WARNING: Decay 1 and 2 from tbar disagree on decay type.")
    if (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: Mismatch').any():      # Want decay 1 and 2 from t to agree
        print("WARNING: Decay 1 and 2 from t disagree on decay type.")
    if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']==df_truth_W['Wdecay_from_t_isHadronicDecay']).any():   # Want exactly one hadronic and one leptonic decay per event
        print("WARNING: Some events have two hadronic or two leptonic decays.")

    # If W from tbar has 0 pdgid, set isHadronicDecay as the opposite of what it is for W from t
    df_truth_W['Wdecay_from_tbar_isHadronicDecay'] = df_truth_W.apply(lambda x : True if x['Wdecay_from_tbar_isHadronicDecay']=='Error: 0' and x['Wdecay_from_t_isHadronicDecay']==False else False if x['Wdecay_from_tbar_isHadronicDecay']=='Error: 0' and x['Wdecay_from_t_isHadronicDecay']==True else x['Wdecay_from_tbar_isHadronicDecay'],axis=1)

    # If W from t has 0 pdgid, set isHadronicDecay as the oppposite of what it is for W from tbar
    df_truth_W['Wdecay_from_t_isHadronicDecay'] = df_truth_W.apply(lambda x : True if x['Wdecay_from_t_isHadronicDecay']=='Error: 0' and x['Wdecay_from_tbar_isHadronicDecay']==False else False if x['Wdecay_from_t_isHadronicDecay']=='Error: 0' and x['Wdecay_from_tbar_isHadronicDecay']==True else x['Wdecay_from_t_isHadronicDecay'],axis=1)

    # One more check for funny business
    if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']==df_truth_W['Wdecay_from_t_isHadronicDecay']).any():   # Want exactly one hadronic and one leptonic decay per event
        print("WARNING: Some events have two hadronic or two leptonic decays.")
    if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: Boson').any() or (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: 0').any() or (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: Mismatch').any():
        print("WARNING: W from tbar stil has an error somewhere. This may mess up data sorting.")
    if (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: Boson').any() or (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: 0').any() or (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: Mismatch').any():
        print("WARNING: W from t stil has an error somewhere. This may mess up data sorting.")


    print("Hadronic and leptonic W's identified.")


    # Make truth df
    df_truth = ak.to_pandas(parton_tree['isDummy'])
    df_truth.rename(columns={'values':'isDummy'},inplace=True)

    # Loop through each of the main variables
    for v in ['pt','eta','phi','m']:
        df_truth['th_'+v] = parton_tree['MC_thad_afterFSR_'+v]    # Append thad from ROOT file
        df_truth['tl_'+v] = parton_tree['MC_tlep_afterFSR_'+v]    # Append tlep from ROOT file 

        # Set wh and wl variables using isHadronicDecay information
        df_truth['wh_'+v] = df_truth_W.apply(lambda x : x['MC_W_from_tbar_afterFSR_'+v] if x['Wdecay_from_tbar_isHadronicDecay']==True else x['MC_W_from_t_afterFSR_'+v],axis=1)    # Sort through W df and find whad
        df_truth['wl_'+v] = df_truth_W.apply(lambda x : x['MC_W_from_t_afterFSR_'+v] if x['Wdecay_from_tbar_isHadronicDecay']==True else x['MC_W_from_tbar_afterFSR_'+v],axis=1)    # Sort through W df and find wlep

        # Append ttbar variables from ROOT file (ONLY for new architecture)
        if ttbar_addon:
            if v=='pt' or v=='m':
                df_truth['ttbar_'+v] = parton_tree['MC_ttbar_afterFSR_'+v]/1000
            else:
                df_truth['ttbar_'+v] = parton_tree['MC_ttbar_afterFSR_'+v]

        # Convert from MeV to GeV
        if v=='pt' or v=='m':
            for p in ['th_','tl_','wh_','wl_']:
                df_truth[p+v] = df_truth[p+v]/1000


    print('Truth dataframe created.')

    return df_truth



# ---------- FUNCTION FOR OBTAINING DATAFRAMES FOR A FILE ---------- #

def getDataframes(name,ttbar_addon,withJetMatch,jn=6):
    """
    Create the jet, other, and truth dataframes from a specified file
    Input: file name (e.g.'1_parton_ejets'), bool for extra ttbar variables, bool for jet matching, number of jets (default=6)
    Output: array of jet dataframes, other dataframe, and truth dataframe
    """

    # Import the root file data
    root_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'_fixed.root')
    reco_tree = root_file['reco'].arrays()
    parton_tree = root_file['parton'].arrays()

    # Remove events with nan y values
    for var in ['MC_thad_afterFSR_y','MC_tlep_afterFSR_y','MC_Wdecay1_from_t_afterFSR_y','MC_Wdecay2_from_t_afterFSR_y','MC_Wdecay1_from_tbar_afterFSR_y','MC_Wdecay2_from_tbar_afterFSR_y']:
        sel = np.invert(np.isnan(parton_tree[var]))
        parton_tree = parton_tree[sel]
        reco_tree = reco_tree[sel]

    # Remove events with met_met < 20 GeV (to match Tao's cut) (not sure KLFitter does this though)
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

    # Get the array of jet dataframes
    df_jets = getJetsData(parton_tree,reco_tree,withJetMatch,jn)

    # Get the other reco dataframe
    df_other = getOtherData(reco_tree)

    # Get the truth dataframe
    df_truth = getTruthData(parton_tree,ttbar_addon)

    # Close root file
    root_file.close()

    return df_jets, df_other, df_truth


# ---------- FUNCTION FOR SAVING H5 FILE ---------- #

def saveH5File(df_jets,df_other,df_truth,save_name):
    """
    Create and save h5 file with the reco and truth variables
    Input: array of jet dataframes, other dataframe, truth dataframe, and save file name (e.g.'1_parton_ejets_addon_ttbar+jetMatch')
    Output: h5 file with jet, other, and truth (th, tl, and possibly ttbar) 4-vectors
    """

    # Creating h5 file for input in the neural network
    f = h5py.File("/data/jchishol/ML_Data/variables_ttbar_"+save_name+".h5","r+")  # "w" means initializing file

    # Create datasets for jets 1 to 6
    for i in range(1,7):
        for v in df_jets[i-1].columns:
            f.create_dataset('j'+str(i)+'_'+v,data=df_jets[i-1][v])

    # Create datasets for other variables
    for v in df_other.columns:
        f.create_dataset(v,data=df_other[v])

    # Data sets for truth variables
    for v in df_truth.columns:        
        f.create_dataset(v,data=df_truth[v])


    print('Saved: variables_ttbar_'+save_name+'.h5')



# ---------- FUNCTION FOR COMBINING H5 FILES ---------- #

def combineH5Files(names,file_nums,save_name):
    """
    Combines the h5 files into one file
    Input: the file name types as an array (e.g. ['parton_ejets_ttbar_addon', 'parton_mjets_ttbar_addon'], the file numbers, and save file name
    Output: saves combined h5 file
    """

    # Create a file to combine the data in
    with h5py.File('/data/jchishol/ML_Data/variables_ttbar_'+save_name+'.h5','w') as h5fw:
        
        current_row = 0   # Keeps track of how many rows of data we have written
        total_len = 0     # Keeps track of how much data we have read
        
        # For each file of that type
        for i in file_nums:
            
            # For each file type (i.e. parton_ejets, parton_mjets, etc)
            for name in names:

                # Read the file
                h5fr = h5py.File('/data/jchishol/ML_Data/variables_ttbar_'+str(i)+'_'+name+'.h5','r')    # Read the file
                
                # Get the file length and add to the total length of data
                dslen = h5fr['j1_isbtag'].shape[0]
                total_len += dslen

                # For each of the variables
                for key in list(h5fr.keys()):

                    # Get the data
                    arr_data = h5fr[key]
                    
                    # If this is the first data file we're looking at, create the dataset from scratch
                    if current_row == 0:
                        h5fw.create_dataset(key,data=arr_data,maxshape=(None,))
                    # Else, resize the dataset length so that it will fit the new data and then append that data
                    else:
                        h5fw[key].resize((total_len,))
                        h5fw[key][current_row:total_len] = arr_data

                # Update the current row
                current_row = total_len

                print('File '+str(i)+' appended.')


# ---------- FUNCTION FOR SAVING XMAXMEAN ---------- #

def saveMaxMean(name,ttbar_addon,withJetMatch):
    """
    Saves numpy array of [max,mean] values for X and Y variables
    Input: file name type, bool for ttbar addon, bool for if it includes jet matches
    Output: two numpy files of [max, mean] values for X and Y variables
    """

    print('Opening file ...')

    # Load data
    f = h5py.File('/data/jchishol/ML_Data/variables_ttbar_'+name+'.h5','r')    

    print('File opened.')

    # Create data frame
    df = pd.DataFrame({'j1_isbtag':np.array(f.get('j1_isbtag'))})
    for key in list(f.keys()):
        df[key] = np.array(f.get(key))

    print('Dataframe created')
    f.close()

    # Initialize arrays of maxmean
    X_maxmean=[]
    Y_maxmean=[]

    # Jets
    for i in range(6):

        # Calculate px and py
        df['j'+str(i+1)+'_px'] = df['j'+str(i+1)+'_pt']*np.cos(df['j'+str(i+1)+'_phi'])
        df['j'+str(i+1)+'_py'] = df['j'+str(i+1)+'_pt']*np.sin(df['j'+str(i+1)+'_phi'])
   
        # Append max and mean
        for v in ['pt','px','py','eta','m','isbtag']:
            X_maxmean.append([df['j'+str(i+1)+'_'+v].abs().max(),df['j'+str(i+1)+'_'+v].mean()])
        
        # Also append isTrue if using the jet matching
        if withJetMatch:
            X_maxmean.append([df['j'+str(i+1)+'_isTrue'].abs().max(),df['j'+str(i+1)+'_isTrue'].mean()])

    print('Jets done')

    # Calculate px and py of lep
    df['lep_px'] = df['lep_pt']*np.cos(df['lep_phi'])
    df['lep_py'] = df['lep_pt']*np.sin(df['lep_phi'])

    print('Calculated lep_px and lep_py.')

    # Calculate sin(met_phi) and cos(met_phi)
    df['met_phi-sin'] = np.sin(df['met_phi'])
    df['met_phi-cos'] = np.cos(df['met_phi'])

    print('Calculated met_phi-sin and met_phi-cos.')

    # Append maxmean for other variables
    for v in ['lep_pt','lep_px','lep_py','lep_eta','met_met','met_phi-sin','met_phi-cos']:
        print('Appending maxmean for '+v)
        X_maxmean.append([df[v].abs().max(),df[v].mean()])

    print('Other done')

    # Save array of X maxmean values
    np.save('X_maxmean_'+name,X_maxmean)
    print('Saved: X_maxmean_'+name+'.npy')

    # Calculate px and py for truth
    particles = ['th_','wh_','tl_','wl_','ttbar_'] if ttbar_addon else ['th_','wh_','tl_','wl_']
    for p in particles:
        df[p+'px'] = df[p+'pt']*np.cos(df[p+'phi'])
        df[p+'py'] = df[p+'pt']*np.sin(df[p+'phi'])

        # Append maxmean for all truth variables
        for v in ['pt','px','py','eta','m']:
            Y_maxmean.append([df[p+v].abs().max(),df[p+v].mean()])

            print('Appended '+p+v)

    # Save Y maxmean arrays
    np.save('Y_maxmean_'+name,Y_maxmean)
    print('Saved: Y_maxmean_'+name+'.npy')



# ---------- MAIN CODE ----------#

# Need to perform the parts separately (otherwise ssh session dies -- not really sure why)

ttbar_addon = True
withJetMatch = True
addon_tag = '_addon_ttbar+jetMatch' if ttbar_addon*withJetMatch else '_addon_ttbar' if ttbar_addon else '_addon_jetMatch' if withJetMatch else ''

# Part 1: create separate h5 files (will need to change the file number each time this is run)
file_num = 0
file_name = '_parton_ejets'
df_jets, df_other, df_truth = getDataframes(str(file_num)+file_name,ttbar_addon,withJetMatch,jn=6)
print(df_jets[0]['isTrue'])
print(df_jets[1]['isTrue'])
print(df_jets[2]['isTrue'])
print(df_jets[3]['isTrue'])
print(df_jets[4]['isTrue'])
print(df_jets[5]['isTrue'])
#saveH5File(df_jets,df_other,df_truth,str(file_num)+file_name+addon_tag)

# Part 2: Put the h5 files together
#combineH5Files(['parton_ejets'+addon_tag,'parton_mjets'+addon_tag],[0,1,2,3,4,6,7],'parton_e+mjets'+addon_tag+'_train')  # Use files 0-4 for training
#combineH5Files(['parton_ejets'+addon_tag],[5],'parton_ejets'+addon_tag+'_test')        # Use file 5 for testing
#combineH5Files(['parton_mjets'+addon_tag],[5],'parton_mjets'+addon_tag+'_test')        # Use file 5 for testing


# Part 3: Create and save numpy array of X and Y max and mean values
saveMaxMean('parton_e+mjets'+addon_tag+'_train',ttbar_addon,withJetMatch)







print ("done :)")
