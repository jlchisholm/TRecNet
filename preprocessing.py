import uproot
import h5py
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import vector

# Import the root file data
root_file = uproot.open('/data/jchishol/mntuple_ttbar_0_parton_ejets.root')
reco_tree = root_file['reco'].arrays()
truth_tree = root_file['parton'].arrays()


# ---------- RECO JETS ---------- #

# Get the total number of events
#num_events = len(reco_tree['eventNumber'])   # This is number of events != number of jets

# Constant for padding the jets
#c = 0.   # Not sure what constant Tao padded with?

# Create temporary jet data frames
#jet_1 = []
#jet_2 = []
#jet_3 = []
#jet_4 = []
#jet_5 = []
#jet_6 = []
#for n in range(num_events):     # Going through each event
    
    # Don't include non-matched events 
#    isMatched = reco_tree['isMatched'][n]
#    if isMatched == 0:
#        continue

    # Get the variables for the event (dividing by 1000 for MeV to GeV)
#    pt = reco_tree['jet_pt'][n]
#    eta = reco_tree['jet_eta'][n]
#    phi = reco_tree['jet_phi'][n]
#    e = reco_tree['jet_e'][n]
#    btagged = reco_tree['jet_btagged'][n]

    # Check that the jets are actually organized highest to lowest pt (this actually only checks that the first is the max)
#    if ak.argmax(pt) != 0:
#        print ('pt values are NOT organized!!!')

    # Append data for the first four jets in the event
#    jet_1.append([pt[0],eta[0],phi[0],e[0],btagged[0]])
#    jet_2.append([pt[1],eta[1],phi[1],e[1],btagged[1]])
#    jet_3.append([pt[2],eta[2],phi[2],e[2],btagged[2]])
#    jet_4.append([pt[3],eta[3],phi[3],e[3],btagged[3]])

    # If there are only four jets, pad jet 5 and 6 with the constant
#    if len(pt)==4:
#        jet_5.append([c,c,c,c,c])
#        jet_6.append([c,c,c,c,c])
    # If there are only five jets, append jet 5 values and pad jet 6 with constant    
#    elif len(pt)==5:
#        jet_5.append([pt[4],eta[4],phi[4],e[4],btagged[4]])
#        jet_6.append([c,c,c,c,c])
    # If there are six jets or more, append jet 5 and 6 values as normal
#    else:
#        jet_5.append([pt[4],eta[4],phi[4],e[4],btagged[4]])
#        jet_6.append([pt[5],eta[5],phi[5],e[5],btagged[5]])

    # Print every once in a while so we know there's progress
#    if n%100000==0:
#        print ('Jet Events Processed: '+str(n))

# Put jet data into data frames
#df_j1 = pd.DataFrame(jet_1,columns=('pt','eta','phi','e','DL1r'))  # ROOT uses btagged, not DL1r, but we'll use Tao's tag for now
#df_j2 = pd.DataFrame(jet_2,columns=('pt','eta','phi','e','DL1r'))
#df_j3 = pd.DataFrame(jet_3,columns=('pt','eta','phi','e','DL1r'))
#df_j4 = pd.DataFrame(jet_4,columns=('pt','eta','phi','e','DL1r'))
#df_j5 = pd.DataFrame(jet_5,columns=('pt','eta','phi','e','DL1r'))
#df_j6 = pd.DataFrame(jet_6,columns=('pt','eta','phi','e','DL1r'))

# Convert from MeV to GeV
#for v in ['pt','e']:
#    df_j1[v] = df_j1[v]/1000
#    df_j2[v] = df_j2[v]/1000
#    df_j3[v] = df_j3[v]/1000
#    df_j4[v] = df_j4[v]/1000
#    df_j5[v] = df_j5[v]/1000
#    df_j6[v] = df_j6[v]/1000

# Calculate mass
#df_j1['m'] = vector.arr({"pt":df_j1['pt'],"phi":df_j1['phi'],"eta":df_j1['eta'],"E":df_j1['e']}).mass
#df_j2['m'] = vector.arr({"pt":df_j2['pt'],"phi":df_j2['phi'],"eta":df_j2['eta'],"E":df_j2['e']}).mass
#df_j3['m'] = vector.arr({"pt":df_j3['pt'],"phi":df_j3['phi'],"eta":df_j3['eta'],"E":df_j3['e']}).mass
#df_j4['m'] = vector.arr({"pt":df_j4['pt'],"phi":df_j4['phi'],"eta":df_j4['eta'],"E":df_j4['e']}).mass
#df_j5['m'] = vector.arr({"pt":df_j5['pt'],"phi":df_j5['phi'],"eta":df_j5['eta'],"E":df_j5['e']}).mass
#df_j6['m'] = vector.arr({"pt":df_j6['pt'],"phi":df_j6['phi'],"eta":df_j6['eta'],"E":df_j6['e']}).mass

# Make an array of jet data frames for easy iteration
#df_jets = [df_j1,df_j2,df_j3,df_j4,df_j5,df_j6]


# ------------ SAVING H5 FILE --------------- #

# ------ X (RECO) VARIABLES ------ #

# Creating h5 file for input in the neural network
#f = h5py.File("test_inputs.h5","w")  # "w" means initializing file
                                     # "a" means append mode
                                     # "r" means read only, file must exist

# Create datasets for jets 1 to 6
#for i in range(1,7):
#    f.create_dataset('j'+str(i)+'_pt',data=df_jets[i-1]['pt'])
#    f.create_dataset('j'+str(i)+'_eta',data=df_jets[i-1]['eta'])
#    f.create_dataset('j'+str(i)+'_phi',data=df_jets[i-1]['phi'])
#    f.create_dataset('j'+str(i)+'_m',data=df_jets[i-1]['m'])
#    f.create_dataset('j'+str(i)+'_DL1r',data=df_jets[i-1]['DL1r'])

# Create dataframe for other variables
#df_other = ak.to_pandas(reco_tree['isMatched'])
#df_other.rename(columns={'values':'isMatched'},inplace=True)
#df_other['lep_pt'] = reco_tree['lep_pt']/1000
#df_other['lep_eta'] = reco_tree['lep_eta']
#df_other['lep_phi'] = reco_tree['lep_phi']
#df_other['met_met'] = reco_tree['met_met']/1000
#df_other['met_phi'] = reco_tree['met_phi']

# Cut rows that are not matched between truth and reco
#indexNames = df_other[df_other['isMatched'] == 0 ].index
#df_other.drop(indexNames,inplace=True)

# Create datasets for other variables
#f.create_dataset('lep_pt',data=df_other['lep_pt'])
#f.create_dataset('lep_eta',data=df_other['lep_eta'])
#f.create_dataset('lep_phi',data=df_other['lep_phi'])
#f.create_dataset('met_met',data=df_other['met_met'])
#f.create_dataset('met_phi',data=df_other['met_phi'])


# ------ Y (TRUTH) VARIABLES ------ #

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

df_truth_W['Wdecay_from_tbar_isHadronicDecay'] = df_truth_W.apply(lambda x : True if x['Wdecay_from_tbar_isHadronicDecay']=='Error: 0' and x['Wdecay_from_t_isHadronicDecay']==False else False if x['Wdecay_from_tbar_isHadronicDecay']=='Error: 0' and x['Wdecay_from_t_isHadronicDecay']==True else x['Wdecay_from_tbar_isHadronicDecay'],axis=1)

df_truth_W['Wdecay_from_t_isHadronicDecay'] = df_truth_W.apply(lambda x : True if x['Wdecay_from_t_isHadronicDecay']=='Error: 0' and x['Wdecay_from_tbar_isHadronicDecay']==False else False if x['Wdecay_from_t_isHadronicDecay']=='Error: 0' and x['Wdecay_from_tbar_isHadronicDecay']==True else x['Wdecay_from_t_isHadronicDecay'],axis=1)

if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']==df_truth_W['Wdecay_from_t_isHadronicDecay']).any():   # Want exactly one hadronic and one leptonic decay per event
    print("WARNING: Some events have two hadronic or two leptonic decays.")
if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: 0').any() or (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: 0').any():
    print("DOUBLE TROUBLE")



# Make truth df
df_truth = ak.to_pandas(truth_tree['isDummy'])
df_truth.rename(columns={'values':'isDummy'},inplace=True)

# Loop through each of the main variables
for v in ['pt','eta','phi','m']:
    df_truth['th_'+v] = truth_tree['MC_thad_afterFSR_'+v]    # Append thad from ROOT file
    df_truth['tl_'+v] = truth_tree['MC_tlep_afterFSR_'+v]    # Append tlep from ROOT file 


    # Note: if W_from_tbar is hadronic, set wh = w_from_tbar, if W_from_tbar is leptonic, set wh = w_from_t, if W_from_tbar is error set wh = w_from_t

    df_truth['wh_'+v] = df_truth_W.apply(lambda x : x['w_from_tbar_'+v] if x['Wdecay_from_tbar_isHadronicDecay']==True else x['w_from_t_'+v],axis=1)    # Sort through W df and find whad
    df_truth['wl_'+v] = df_truth_W.apply(lambda x : x['w_from_tbar_'+v] if x['Wdecay_from_tbar_isHadronicDecay']==False else x['w_from_t_'+v],axis=1)   # Sort through W df and find wlep

    # Convert from MeV to GeV
    if v=='pt' or v=='m':
        df_truth['th_'+v] = df_truth['th_'+v]/1000
        df_truth['tl_'+v] = df_truth['tl_'+v]/1000
        df_truth['wh_'+v] = df_truth['wh_'+v]/1000
        df_truth['wl_'+v] = df_truth['wl_'+v]/1000

# Drop dummy truth events
indexNames = df_truth[df_truth['isDummy'] == 1 ].index
df_truth.drop(indexNames,inplace=True)

# Data sets for truth variables
#for v in ['pt','eta','phi','m']:
#    f.create_dataset('th_'+v,data=df_truth['th_'+v])
#    f.create_dataset('tl_'+v,data=df_truth['tl_'+v])
#    f.create_dataset('wh_'+v,data=df_truth['wh_'+v])
#    f.create_dataset('wl_'+v,data=df_truth['wl_'+v])


#print('h5 file created.')



# --------- Create {X,Y}_maxmean1.npy Files ----------- #

# --- X --- #

# Initialize array of maxmean
#X_maxmean=[]

# Append maxmean for each jet
#for df in df_jets:
    
    # Calculate px and py
#    df['px'] = df['pt']*np.cos(df['phi'])
#    df['py'] = df['pt']*np.sin(df['phi'])

#    for v in ['pt','px','py','eta','m','DL1r']:
#        X_maxmean.append([df[v].max(),df[v].mean()])

# Calculate px and py for lep
#df_other['lep_px'] = df_other['lep_pt']*np.cos(df_other['lep_phi'])
#df_other['lep_py'] = df_other['lep_pt']*np.sin(df_other['lep_phi'])

# Calculate sin(met_phi) and cos(met_phi)
#df_other['met_phi-sin'] = np.sin(df_other['met_phi'])
#df_other['met_phi-cos'] = np.cos(df_other['met_phi'])

# Append maxmean for other variables
#for v in ['lep_pt','lep_px','lep_py','lep_eta','met_met','met_phi-sin','met_phi-cos']:
#    X_maxmean.append([df_other[v].max(),df_other[v].mean()])

#print('X maxmean data collected.')

# --- Y --- #

# Initialize array of maxmean
#Y_maxmean=[]

# Calculate px and py for variables
#for p in ['th_','wh_','tl_','wl_']:
#    df_truth[p+'px'] = df_truth[p+'pt']*np.cos(df_truth[p+'phi'])
#    df_truth[p+'py'] = df_truth[p+'pt']*np.sin(df_truth[p+'phi'])

    # Append maxmean for all truth variables
#    for v in ['pt','px','py','eta','m']:
#        if p=='th_' and v=='eta':
#            indexNames = df_truth[df_truth[p+v] >= -20 ].index.tolist()   # Outliers were throwing off the mean (might want to cut events instead?)
#            mean = df_truth[p+v].iloc[indexNames].mean(axis=0)
#            Y_maxmean.append([df_truth[p+v].max(),mean])
#        else:
#            Y_maxmean.append([df_truth[p+v].max(),df_truth[p+v].mean()])


#print('Y maxmean data collected.')

# Save maxmean arrays
#np.save('X_maxmean_new',X_maxmean)
#np.save('Y_maxmean_new',Y_maxmean)




#print(df_j1)
#print('---------')
#print(df_other)
#print('---------')
#print(df_truth)
#print('---------')
#print(X_maxmean)
#print('---------')
#print(Y_maxmean)




print ("done :)")
