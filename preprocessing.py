import uproot
import h5py
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import vector



# ---------- FUNCTION FOR OBTAINING JET DATA ---------- #

# Creates an array of 6 dataframes for the 6 jets
# Input: reco level tree from ROOT file, number of jets to have per event (default is 6)
def getJetsData(reco_tree,jn=6):

    # Get the total number of events
    num_events = len(reco_tree['eventNumber'])
    print('Number of events: '+str(num_events))

    # Constant for padding the jets
    c = 0.
    b = -1.

    # Create temporary jet arrays
    jets = [[] for _ in range(jn)]
    for n in range(num_events):     # Going through each event

        # Get the variables for the jets in this event
        pt = reco_tree['jet_pt'][n]
        eta = reco_tree['jet_eta'][n]
        phi = reco_tree['jet_phi'][n]
        e = reco_tree['jet_e'][n]
        btag = reco_tree['jet_btagged'][n]

        # Get the number of jets in this event
        n_jets = reco_tree['jet_n'][n]

        # Check that the jets are actually organized highest to lowest pt and reorganize if necessary (hasn't been an issue yet)
        if not np.array_equal(np.array(sorted(pt,reverse=True)),np.array(pt)):
            print('Jet pt values are NOT sorted highest to lowest. Attempting to re-order properly...')
            print(pt)
            
            # Reorder jets from highest to lowest pt
            all_var = zip(pt,eta,phi,e,btag)
            all_var_by_pt = sorted(all_var,reverse=True)
            pt = [pt for pt,_,_,_,_ in all_var_by_pt]
            eta = [eta for _,eta,_,_,_ in all_var_by_pt]
            phi = [phi for _,_,phi,_,_ in all_var_by_pt]
            e = [e for _,_,_,e,_ in all_var_by_pt]
            btag = [btag for _,_,_,_,btag in all_var_by_pt]

        # Append data for the jets
        for i in range(jn):
            if i<=n_jets:  # Append like normal for however many jets are in the event
                jets[i].append([pt[i],eta[i],phi[i],e[i],btag[i]])
            else:          # Otherwise pad the jets with constants
                jets[i].append([c,c,c,c,b])

        # Print every once in a while so we know there's progress
        if (n+1)%100000==0 or n==num_events-1:
            print ('Jet Events Processed: '+str(n+1))

    # Create an array of jet dataframes
    df_jets = [[] for _ in range(6)]
    for i in range(len(df_jets)):
        df_jets[i] = pd.DataFrame(jets[i],columns=('pt','eta','phi','e','isbtag'))

        # Convert from MeV to GeV
        for v in ['pt','e']:
            df_jets[i][v] = df_jets[i][v]/1000

        # Calculate mass
        df_jets[i]['m'] = vector.arr({"pt":df_jets[i]['pt'],"phi":df_jets[i]['phi'],"eta":df_jets[i]['eta'],"E":df_jets[i]['e']}).mass

        # Print a progress statement
        print ('Jet '+str(i+1)+' dataframe created.')

    return df_jets


# ---------- FUNCTION FOR OBTAINING OTHER DATA ---------- #

# Create a dataframe for the truth variables
# Input: reco level tree from ROOT file
def getOtherData(reco_tree):

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

# Create a dataframe for the truth variables
# Input: truth level tree from ROOT file, true or false for added ttbar variables
def getTruthData(truth_tree,ttbar_addon):

    # Create temporary df for W truth variables
    df_truth_W = ak.to_pandas(truth_tree['isDummy'])
    df_truth_W.rename(columns={'values':'isDummy'},inplace=True)

    # Need to calculate the W 4-vectors from the decay products due to a problem with the mntuple!

    #decay_vec = []
    # Append the pdgid values for each of the four W decay products
    for par in ['Wdecay1_from_tbar_','Wdecay1_from_t_','Wdecay2_from_tbar_','Wdecay2_from_t_']:
        df_truth_W[par+'pdgid'] = truth_tree['MC_'+par+'afterFSR_pdgid']

        # Get four vector of the W decay product (vector.arr not working properly at the moment ... need to go event by event :( )
        #decay_vec.append(vector.arr({"pt":truth_tree['MC_'+par+'afterFSR_pt'],"phi":truth_tree['MC_'+par+'afterFSR_phi'],"eta":truth_tree['MC_'+par+'afterFSR_eta'],"mass":truth_tree['MC_'+par+'afterFSR_m']}))   

    # Get 4-vector values for the W's into the dataframe
    for t in ['t','tbar']:

        print('Calculating 4-vectors for '+t+' ...')

        # Create lists for the variables
        W_pt = []
        W_phi = []
        W_eta = []
        W_m = []

        # Run through each event
        num_events = len(df_truth_W['isDummy'])
        for i in range(num_events):

            # Get W decay vectors
            # ('Wdecay2_from_t_eta' is full of zeroes, so we need to approximate with y for now)
            if t=='t':
                Wdecay2_vec = vector.obj(pt=truth_tree['MC_Wdecay2_from_'+t+'_afterFSR_pt'][i],phi=truth_tree['MC_Wdecay2_from_'+t+'_afterFSR_phi'][i],eta=truth_tree['MC_Wdecay2_from_'+t+'_afterFSR_y'][i],mass=truth_tree['MC_Wdecay2_from_'+t+'_afterFSR_m'][i])
            else:
                Wdecay2_vec = vector.obj(pt=truth_tree['MC_Wdecay2_from_'+t+'_afterFSR_pt'][i],phi=truth_tree['MC_Wdecay2_from_'+t+'_afterFSR_phi'][i],eta=truth_tree['MC_Wdecay2_from_'+t+'_afterFSR_eta'][i],mass=truth_tree['MC_Wdecay2_from_'+t+'_afterFSR_m'][i])

            Wdecay1_vec = vector.obj(pt=truth_tree['MC_Wdecay1_from_'+t+'_afterFSR_pt'][i],phi=truth_tree['MC_Wdecay1_from_'+t+'_afterFSR_phi'][i],eta=truth_tree['MC_Wdecay1_from_'+t+'_afterFSR_eta'][i],mass=truth_tree['MC_Wdecay1_from_'+t+'_afterFSR_m'][i])

            # Add to get W vector
            W_vec = Wdecay1_vec + Wdecay2_vec

            # Append vector values to list
            W_pt.append(W_vec.pt)
            W_phi.append(W_vec.phi)
            W_eta.append(W_vec.eta)
            W_m.append(W_vec.mass)

        print('Appending W values to dataframe ...')

        # Put values in dataframe
        df_truth_W['w_from_'+t+'_pt'] = W_pt
        df_truth_W['w_from_'+t+'_phi'] = W_phi
        df_truth_W['w_from_'+t+'_eta'] = W_eta
        df_truth_W['w_from_'+t+'_m'] = W_m


    # Add 4-vectors of two decay products from same W
    #W_from_tbar_vec = decay_vec[0] + decay_vec[2]
    #W_from_t_vec = decay_vec[1] + decay_vec[3]

    # Append pt, eta, phi, and m to the data frame for both W's
    #df_truth_W['w_from_tbar_pt'] = W_from_tbar_vec.pt
    #df_truth_W['w_from_tbar_eta'] = W_from_tbar_vec.eta
    #df_truth_W['w_from_tbar_phi'] = W_from_tbar_vec.phi
    #df_truth_W['w_from_tbar_m'] = W_from_tbar_vec.mass
    #df_truth_W['w_from_tbar_y'] = W_from_tbar_vec.y

    #df_truth_W['w_from_t_pt'] = W_from_tbar_vec.pt
    #df_truth_W['w_from_t_eta'] = W_from_tbar_vec.eta
    #df_truth_W['w_from_t_phi'] = W_from_tbar_vec.phi
    #df_truth_W['w_from_t_m'] = W_from_tbar_vec.mass
    #df_truth_W['w_from_t_y'] = W_from_tbar_vec.y


    # Define some helpful ranges
    had_range = list(range(1,9))              # Quarks have pdgid 1 to 8
    lep_range = list(range(11,19))            # Leptons have pdgid 11 to 18
    bos_range = list(range(21,26))+[9,37]     # Bosons have pdgid 21 to 25, and 9, and 37

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


    print("Hadronic and leptonic W's identified and sorted.")


    # Make truth df
    df_truth = ak.to_pandas(truth_tree['isDummy'])
    df_truth.rename(columns={'values':'isDummy'},inplace=True)

    # Loop through each of the main variables
    for v in ['pt','eta','phi','m']:
        df_truth['th_'+v] = truth_tree['MC_thad_afterFSR_'+v]    # Append thad from ROOT file
        df_truth['tl_'+v] = truth_tree['MC_tlep_afterFSR_'+v]    # Append tlep from ROOT file 

        # Set wh and wl variables using isHadronicDecay information
        df_truth['wh_'+v] = df_truth_W.apply(lambda x : x['w_from_tbar_'+v] if x['Wdecay_from_tbar_isHadronicDecay']==True else x['w_from_t_'+v],axis=1)    # Sort through W df and find whad
        df_truth['wl_'+v] = df_truth_W.apply(lambda x : x['w_from_t_'+v] if x['Wdecay_from_tbar_isHadronicDecay']==True else x['w_from_tbar_'+v],axis=1)    # Sort through W df and find wlep

        # Append ttbar variables from ROOT file (ONLY for new architecture)
        if ttbar_addon:
            if v=='pt' or v=='m':
                df_truth['ttbar_'+v] = truth_tree['MC_ttbar_afterFSR_'+v]/1000
            else:
                df_truth['ttbar_'+v] = truth_tree['MC_ttbar_afterFSR_'+v]

        # Convert from MeV to GeV
        if v=='pt' or v=='m':
            for p in ['th_','tl_','wh_','wl_']:
                df_truth[p+v] = df_truth[p+v]/1000


    print('Truth dataframe created.')

    return df_truth


# ---------- FUNCTION FOR OBTAINING DATAFRAMES FOR A FILE ---------- #

# Create the jet, other, and truth dataframes from a specified file
# Input: file name, true or false for extra ttbar variables
def getDataframes(name,ttbar_addon):

    # Import the root file data
    root_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'.root')
    reco_tree = root_file['reco'].arrays()
    truth_tree = root_file['parton'].arrays()

    # Remove events with nan y values
    for var in ['MC_thad_afterFSR_y','MC_tlep_afterFSR_y','MC_Wdecay1_from_t_afterFSR_y','MC_Wdecay2_from_t_afterFSR_y','MC_Wdecay1_from_tbar_afterFSR_y','MC_Wdecay2_from_tbar_afterFSR_y']:
        sel = np.invert(np.isnan(truth_tree[var]))
        truth_tree = truth_tree[sel]
        reco_tree = reco_tree[sel]

    # Remove events with met_met < 20 (to match Tao's cut) (not sure KLFitter does this though)
    sel = reco_tree['met_met']/1000 >= 20
    truth_tree = truth_tree[sel]
    reco_tree = reco_tree[sel]

    # Remove isDummy events and !isMatched events
    sel = reco_tree['isMatched']==1
    truth_tree = truth_tree[sel]
    reco_tree = reco_tree[sel]
    sel = truth_tree['isDummy']==0
    truth_tree = truth_tree[sel]
    reco_tree = reco_tree[sel]

    # Get the array of jet dataframes
    df_jets = getJetsData(reco_tree)

    # Get the other reco dataframe
    df_other = getOtherData(reco_tree)

    # Get the truth dataframe
    df_truth = getTruthData(truth_tree,ttbar_addon)

    # Close root file
    root_file.close()

    return df_jets, df_other, df_truth


# ---------- FUNCTION FOR SAVING H5 FILE ---------- #

# Create and save h5 file with the reco and truth variables
# Input: array of jet dataframes, other dataframe, truth dataframe, file number, and name
def saveH5File(df_jets,df_other,df_truth,n,name):

    # Creating h5 file for input in the neural network
    f = h5py.File("/data/jchishol/ML_Data/variables_ttbar_"+str(n)+name+".h5","r+")  # "w" means initializing file

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


    print('Saved: variables_ttbar_'+str(n)+name+'.h5')


# ---------- FUNCTION FOR SAVING XMAXMEAN ---------- #

# Saves numpy array of [max,mean] values for X and Y variables
# Input: file name type
def saveMaxMean(name):

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
    for p in ['th_','wh_','tl_','wl_','ttbar_']:
        df[p+'px'] = df[p+'pt']*np.cos(df[p+'phi'])
        df[p+'py'] = df[p+'pt']*np.sin(df[p+'phi'])

        # Append maxmean for all truth variables
        for v in ['pt','px','py','eta','m']:
            Y_maxmean.append([df[p+v].abs().max(),df[p+v].mean()])

            print('Appended '+p+v)

    # Save Y maxmean arrays
    np.save('Y_maxmean_'+name,Y_maxmean)
    print('Saved: Y_maxmean_'+name+'.npy')


# ---------- FUNCTION FOR COMBINING H5 FILES ---------- #

# Combines the h5 files into one file
# Input: the file name types as an array (e.g. ['parton_ejets', 'parton_mjets'], the file numbers, save file name
def combineH5Files(names,file_nums,f_name):

    # Set the name the new file will be called based on what's going into it
    #if len(names)==2:
    #    f_name = 'parton_e+mjets'
    #elif len(names)==1:
    #    f_name = names[0]
    #else:
    #    sys.exit('Too many or too little file names.')

    # Create a file to combine the data in
    with h5py.File('/data/jchishol/ML_Data/variables_ttbar_'+f_name+'.h5','w') as h5fw:
        
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





# ---------- MAIN CODE ----------#

# Need to perform the parts separately (otherwise ssh session dies -- not really sure why)

# Part 1: create separate h5 files (will need to change the file number each time this is run)
file_num = 7
file_name = '_parton_mjets'
df_jets, df_other, df_truth = getDataframes(str(file_num)+file_name,True)
saveH5File(df_jets,df_other,df_truth,file_num,file_name+'_ttbar_addon')


# Part 2: Put the h5 files together
combineH5Files(['parton_ejets_ttbar_addon','parton_mjets_ttbar_addon'],[0,1,2,3,4,6,7],'parton_e+mjets_ttbar_addon_train')  # Use files 0-4 for training
combineH5Files(['parton_ejets_ttbar_addon'],[5],'parton_ejets_ttbar_addon_test')        # Use file 5 for testing
combineH5Files(['parton_mjets_ttbar_addon'],[5],'parton_mjets_ttbar_addon_test')        # Use file 5 for testing


# Part 3: Create and save numpy array of X and Y max and mean values
saveMaxMean('parton_e+mjets_ttbar_addon_train')







print ("done :)")
