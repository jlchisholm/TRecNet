import uproot
import h5py
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import vector


# ---------- FUNCTION FOR OBTAINING JET DATA ---------- #

# Creates an array of 6 dataframes for the 6 jets
# Input: reco level tree from ROOT file
def getJetsData(reco_tree):

    # Get the total number of events
    num_events = len(reco_tree['eventNumber'])   # This is number of events != number of jets

    # Constant for padding the jets
    c = 0.   # I think this is the constant Tao padded with?

    # Create temporary jet arrays
    jets = [[] for _ in range(6)]
    for n in range(num_events):     # Going through each event

        # Don't include non-matched events 
        isMatched = reco_tree['isMatched'][n]
        if isMatched == 0:
            continue

        # Get the variables for the event
        pt = reco_tree['jet_pt'][n]
        eta = reco_tree['jet_eta'][n]
        phi = reco_tree['jet_phi'][n]
        e = reco_tree['jet_e'][n]
        btagged = reco_tree['jet_btagged'][n]
        n_jets = reco_tree['jet_n'][n]

        # Check that the jets are actually organized highest to lowest pt (this actually only checks that the first is the max)
        if ak.argmax(pt) != 0:
            print ('pt values are NOT organized!!!')

        # Append data for the first four jets in the event
        jets[0].append([pt[0],eta[0],phi[0],e[0],btagged[0]])
        jets[1].append([pt[1],eta[1],phi[1],e[1],btagged[1]])
        jets[2].append([pt[2],eta[2],phi[2],e[2],btagged[2]])
        jets[3].append([pt[3],eta[3],phi[3],e[3],btagged[3]])

        # If there are only four jets, pad jet 5 and 6 with the constant
        if n_jets==4:
            jets[4].append([c,c,c,c,c])
            jets[5].append([c,c,c,c,c])
        # If there are only five jets, append jet 5 values and pad jet 6 with constant    
        elif n_jets==5:
            jets[4].append([pt[4],eta[4],phi[4],e[4],btagged[4]])
            jets[5].append([c,c,c,c,c])
        # If there are six jets or more, append jet 5 and 6 values as normal
        else:
            jets[4].append([pt[4],eta[4],phi[4],e[4],btagged[4]])
            jets[5].append([pt[5],eta[5],phi[5],e[5],btagged[5]])

        # Print every once in a while so we know there's progress
        if n%100000==0:
            print ('Jet Events Processed: '+str(n))

    # Create an array of jet dataframes
    df_jets = [[] for _ in range(6)]
    for i in range(len(df_jets)):
        df_jets[i] = pd.DataFrame(jets[i],columns=('pt','eta','phi','e','DL1r')) # ROOT uses btagged, not D1r, but we'll use Tao's tag for now

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
    df_other['lep_pt'] = reco_tree['lep_pt']/1000
    df_other['lep_eta'] = reco_tree['lep_eta']
    df_other['lep_phi'] = reco_tree['lep_phi']
    df_other['met_met'] = reco_tree['met_met']/1000
    df_other['met_phi'] = reco_tree['met_phi']

    # Cut rows that are not matched between truth and reco
    indexNames = df_other[df_other['isMatched'] == 0 ].index
    df_other.drop(indexNames,inplace=True)
    
    print('Other dataframe created.')

    return df_other

# ---------- FUNCTION FOR OBTAINING TRUTH DATA ---------- #

# Create a dataframe for the truth variables
# Input: truth level tree from ROOT file
def getTruthData(truth_tree):

    # Create temporary df for W truth variables
    df_truth_W = ak.to_pandas(truth_tree['isDummy'])
    df_truth_W.rename(columns={'values':'isDummy'},inplace=True)

    decay_vec = []
    # Append the pdgid values for each of the four W decay products
    for par in ['Wdecay1_from_tbar_','Wdecay1_from_t_','Wdecay2_from_tbar_','Wdecay2_from_t_']:
        df_truth_W[par+'pdgid'] = truth_tree['MC_'+par+'afterFSR_pdgid']

        # Get four vector of the W decay product
        decay_vec.append(vector.arr({"pt":truth_tree['MC_'+par+'afterFSR_pt'],"phi":truth_tree['MC_'+par+'afterFSR_phi'],"eta":truth_tree['MC_'+par+'afterFSR_eta'],"mass":truth_tree['MC_'+par+'afterFSR_m']}))

    # Add 4-vectors of two decay products from same W
    W_from_tbar_vec = decay_vec[0] + decay_vec[2]
    W_from_t_vec = decay_vec[1] + decay_vec[3]

    # Append pt, eta, phi, and m to the data frame for both W's
    df_truth_W['w_from_tbar_pt'] = W_from_tbar_vec.pt
    df_truth_W['w_from_tbar_eta'] = W_from_tbar_vec.eta
    df_truth_W['w_from_tbar_phi'] = W_from_tbar_vec.phi
    df_truth_W['w_from_tbar_m'] = W_from_tbar_vec.mass

    df_truth_W['w_from_t_pt'] = W_from_tbar_vec.pt
    df_truth_W['w_from_t_eta'] = W_from_tbar_vec.eta
    df_truth_W['w_from_t_phi'] = W_from_tbar_vec.phi
    df_truth_W['w_from_t_m'] = W_from_tbar_vec.mass

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

        # Convert from MeV to GeV
        if v=='pt' or v=='m':
            for p in ['th_','tl_','wh_','wl_']:
                df_truth[p+v] = df_truth[p+v]/1000

    # Drop dummy truth events
    indexNames = df_truth[df_truth['isDummy'] == 1 ].index
    df_truth.drop(indexNames,inplace=True)

    print('Truth dataframe created.')

    return df_truth


# ---------- FUNCTION FOR OBTAINING DATAFRAMES FOR A FILE ---------- #

# Create the jet, other, and truth dataframes from a specified file
# Input: file name
def getDataframes(name):

    # Import the root file data
    root_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'.root')
    reco_tree = root_file['reco'].arrays()
    truth_tree = root_file['parton'].arrays()

    # Get the array of jet dataframes
    df_jets = getJetsData(reco_tree)

    # Get the other reco dataframe
    df_other = getOtherData(reco_tree)

    # Get the truth dataframe
    df_truth = getTruthData(truth_tree)

    # Close root file
    root_file.close()

    return df_jets, df_other, df_truth


# ---------- FUNCTION FOR SAVING H5 FILE ---------- #

# Create and save h5 file with the reco and truth variables
# Input: array of jet dataframes, other dataframe, and truth dataframe
def saveH5File(df_jets,df_other,df_truth,n):

    # Creating h5 file for input in the neural network
    f = h5py.File("/data/jchishol/variables_ttbar_"+str(n)+"_parton_ejets.h5","w")  # "w" means initializing file

    # Create datasets for jets 1 to 6
    for i in range(1,7):
        for v in ['pt','eta','phi','m','DL1r']:
            f.create_dataset('j'+str(i)+'_'+v,data=df_jets[i-1][v])

    # Create datasets for other variables
    f.create_dataset('lep_pt',data=df_other['lep_pt'])
    f.create_dataset('lep_eta',data=df_other['lep_eta'])    
    f.create_dataset('lep_phi',data=df_other['lep_phi'])
    f.create_dataset('met_met',data=df_other['met_met'])
    f.create_dataset('met_phi',data=df_other['met_phi'])

    # Data sets for truth variables
    for v in ['pt','eta','phi','m']:
        f.create_dataset('th_'+v,data=df_truth['th_'+v])
        f.create_dataset('tl_'+v,data=df_truth['tl_'+v])
        f.create_dataset('wh_'+v,data=df_truth['wh_'+v])
        f.create_dataset('wl_'+v,data=df_truth['wl_'+v])

    print('Saved: variables_ttbar_'+str(n)+'_parton_ejets.h5')


# ---------- FUNCTION FOR SAVING XMAXMEAN ---------- #

# Saves numpy array of [max,mean] values for X and Y variables
def saveMaxMean():

    # Load data
    f = h5py.File('/data/jchishol/Jenna_Data/variables_ttbar_parton_ejets.h5','r')    

    # Create data frame
    df = pd.DataFrame({'j1_DL1r':np.array(f.get('j1_DL1r'))})
    for key in list(f.keys()):
        df[key] = np.array(f.get(key))

    # Initialize arrays of maxmean
    X_maxmean=[]
    Y_maxmean=[]

    # Jets
    for i in range(6):

        # Calculate px and py
        df['j'+str(i+1)+'_px'] = df['j'+str(i+1)+'_pt']*np.cos(df['j'+str(i+1)+'_phi'])
        df['j'+str(i+1)+'_py'] = df['j'+str(i+1)+'_pt']*np.sin(df['j'+str(i+1)+'_phi'])
   
        # Append max and mean
        for v in ['pt','px','py','eta','m','DL1r']:
            X_maxmean.append([df['j'+str(i+1)+'_'+v].max(),df['j'+str(i+1)+'_'+v].mean()])

    # Calculate px and py of lep
    df['lep_px'] = df['lep_pt']*np.cos(df['lep_phi'])
    df['lep_py'] = df['lep_pt']*np.sin(df['lep_phi'])

    # Calculate sin(met_phi) and cos(met_phi)
    df['met_phi-sin'] = np.sin(df['met_phi'])
    df['met_phi-cos'] = np.cos(df['met_phi'])

    # Append maxmean for other variables
    for v in ['lep_pt','lep_px','lep_py','lep_eta','met_met','met_phi-sin','met_phi-cos']:
        X_maxmean.append([df[v].max(),df[v].mean()])

    # Save array of X maxmean values
    np.save('X_maxmean_new',X_maxmean)
    print('Saved: X_maxmean_new.npy')

    # Calculate px and py for truth
    for p in ['th_','wh_','tl_','wl_']:
        df[p+'px'] = df[p+'pt']*np.cos(df[p+'phi'])
        df[p+'py'] = df[p+'pt']*np.sin(df[p+'phi'])

        # Append maxmean for all truth variables
        for v in ['pt','px','py','eta','m']:
            if v=='eta':
                indexNames = df[abs(df[p+v]) <= 20 ].index.tolist()   # Outliers were throwing off the mean (might want to cut events instead?)
                mean = df[p+v].iloc[indexNames].mean(axis=0)
                Y_maxmean.append([df[p+v].max(),mean])
            else:
                Y_maxmean.append([df[p+v].max(),df[p+v].mean()])

            print('Appended '+p+v)

    # Save maxmean arrays
    np.save('Y_maxmean_new',Y_maxmean)
    print('Saved: Y_maxmean_new.npy')


# ---------- FUNCTION FOR TURNING ROOT DATA TO H5 FILE ---------- #

# Takes data from designated root files and creates an h5 file for machine learning
# Input designated number of files to process
#def rootDataToH5(nf):

    # Get dataframes for each of the files
#    df_jets_list = [[] for _ in range(nf)]
#    df_other_list = [[] for _ in range(nf)]
#    df_truth_list = [[] for _ in range(nf)]
#    for i in range(nf):
#        print('Processing file: '+str(i))
#        df_jets_list[i], df_other_list[i], df_truth_list[i] = getDataframes(str(i)+'_parton_ejets')

    # Initialize concatenated data frames
#    df_jets = [[] for _ in range(6)]
#    for i in range(6):
#        df_jets[i] = pd.concat([df_jets_list[0][i],df_jets_list[1][i]])   # Initialize jets 
#    df_other = pd.concat([df_other_list[0],df_other_list[1]])             # Initialize other
#    df_truth = pd.concat([df_truth_list[0],df_truth_list[1]])             # Initialize truth
#
    # Concatenate the rest of the data
#    for j in range(2,nf+1):
#        for i in range(6):
#            df_jets[i] = pd.concat([df_jets[i],df_jets_list[j][i]])   # Concatenate jets
#        df_other = pd.concat([df_other,df_other_list[j]])             # Concatenate other
#        df_truth = pd.concat([df_truth,df_truth_list[j]])             # Concatenate truth

#    print('Dataframes concatenated.')

#    print(df_truth)

    # Create and save an h5 file of the data
#    saveH5File(df_jets,df_other,df_truth)


# ---------- MAIN CODE ----------#

# Create h5 file
#rootDataToH5(7)

# Separate h5 files?
#file_num = 7
#df_jets, df_other, df_truth = getDataframes(str(file_num)+'_parton_ejets')
#print('Made it here!')
#saveH5File(df_jets,df_other,df_truth,file_num)


# Put h5 files together?
#with h5py.File('/data/jchishol/Jenna_Data/variables_ttbar_parton_ejets.h5','w') as h5fw:
#    current_row = 0
#    total_len = 0
#    for i in range(8):
#        h5fr = h5py.File('/data/jchishol/variables_ttbar_'+str(i)+'_parton_ejets.h5','r')    # Read the file
#        dslen = h5fr['j1_DL1r'].shape[0]
#        total_len += dslen
#        for key in list(h5fr.keys()):
#            arr_data = h5fr[key]
#            if i == 0:
#                h5fw.create_dataset(key,data=arr_data,maxshape=(None,))
#            else:
#                h5fw[key].resize((total_len,))
#                h5fw[key][current_row:total_len] = arr_data
#        current_row = total_len
#        print('File '+str(i)+' appended.')




# Create and save numpy array of X and Y max and mean values
saveMaxMean()







print ("done :)")
