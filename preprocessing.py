import uproot
import h5py
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib
#matplotlib.use('GTK3Agg')   # need this one if you want plots displayed
import vector
from scipy.optimize import linear_sum_assignment

# Define some helpful ranges
had_range = list(range(1,9))+list(range(-8,0))                     # Quarks have pdgid 1 to 8 (negatives indicate antiparticles)
lep_range = list(range(11,19))+list(range(-18,10))                 # Leptons have pdgid 11 to 18
bos_range = list(range(21,26))+list(range(-25,20))+[9,37,-9,-37]   # Bosons have pdgid 21 to 25, and 9, and 37



# ---------- FUNCTION FOR OBTAINING JET DATA ---------- #

def getJetsData(reco_tree,jn=6):
    """
    Creates an array of dataframes for the jets
    Input: reco level tree from ROOT file, bool for if you want to match jets, number of jets <jn> want to have per event (default is 6)
    Output: an array of <jn> dataframes (one for each of the jets)
    """

    # Create an array of <jn> jet dataframes
    df_jets = [pd.DataFrame() for _ in range(jn)]
    
    # Pad pt, eta, phi, e with 0's for events with less than <jn> jets, and add these variables to each of the jet dataframes
    for v in ['pt','eta','phi','e']:
        padded_vars = np.asarray(ak.fill_none(ak.pad_none(reco_tree['jet_'+v], jn, clip=True), 0.))
        for j, df in enumerate(df_jets):
            df[v] = padded_vars[:,j]

    # Pad btag with -1.'s
    padded_btags = np.asarray(ak.fill_none(ak.pad_none(reco_tree['jet_btagged'], jn, clip=True), -1))

    # Finish off the dataframes
    for j, df in enumerate(df_jets):

        # Put m in the dataframes and get rid of e (which we won't really need)
        df['m'] = df.apply(lambda row : vector.obj(pt=row['pt'],eta=row['eta'],phi=row['phi'],E=row['e']).mass,axis=1)
        df.drop(columns=['e'],inplace=True)

        # Include btags
        df['isbtag'] = padded_btags[:,j]

        # Convert from MeV to GeV
        df['pt'] = df['pt']/1000
        df['m'] = df['m']/1000


    print('Jet dataframes created.')

    return df_jets


# ---------- FUNCTION FOR OBTAINING OTHER DATA ---------- #

def getOtherData(reco_tree):
    """
    Create a dataframe for the other (lep, met) variables
    Input: reco level tree from ROOT file 
    Output: dataframe for the other (lep, met) variables
    """

    # Create dataframe for other variables (note divide lep_pt and met_met by 1000 to convert from MeV to GeV)
    keys = ['lep_pt','lep_eta','lep_phi','met_met','met_phi']
    df_other = ak.to_pandas({key:reco_tree[key]/1000 if key=='lep_pt' or key=='met_met' else reco_tree[key] for key in keys})

    print('Other dataframe created.')

    return df_other


# ---------- FUNCTION FOR OBTAINING TRUTH DATA ---------- #

def getTruthData(parton_tree,reco_tree,ttbar_addon,withJetMatch,jn=6):
    """
    Create a dataframe for the truth variables
    Input: parton level tree from ROOT file, true or false for added ttbar variables
    Output: dataframe for the truth variables
    """

    # Create temporary df for W truth variables (to help sort the hadronic and leptonic)
    W_keys = ['MC_Wdecay1_from_tbar_afterFSR_pdgid','MC_Wdecay1_from_t_afterFSR_pdgid','MC_Wdecay2_from_tbar_afterFSR_pdgid','MC_Wdecay2_from_t_afterFSR_pdgid']
    for par in ['MC_W_from_t_afterFSR_','MC_W_from_tbar_afterFSR_']:
        for var in ['pt','eta','y','phi','m','E']:
            W_keys.append(par+var)
    df_truth_W = ak.to_pandas({key:parton_tree[key] for key in W_keys})
    
    # Determine if the W decay from tbar is a hadronic decay
    df_truth_W['Wdecay_from_tbar_isHadronicDecay'] = df_truth_W.apply(lambda x : True if abs(x['MC_Wdecay1_from_tbar_afterFSR_pdgid']) in had_range and abs(x['MC_Wdecay2_from_tbar_afterFSR_pdgid']) in had_range else False if abs(x['MC_Wdecay1_from_tbar_afterFSR_pdgid']) in lep_range and abs(x['MC_Wdecay2_from_tbar_afterFSR_pdgid']) in lep_range  else 'Error: Boson' if abs(x['MC_Wdecay1_from_tbar_afterFSR_pdgid']) in bos_range or abs(x['MC_Wdecay2_from_tbar_afterFSR_pdgid']) in bos_range else 'Error: 0' if abs(x['MC_Wdecay1_from_tbar_afterFSR_pdgid'])==0 or abs(x['MC_Wdecay2_from_tbar_afterFSR_pdgid'])==0 else 'Error: Mismatch' ,axis=1)

    # Determine if the W decay from t is a hadronic decay
    df_truth_W['Wdecay_from_t_isHadronicDecay'] = df_truth_W.apply(lambda x : True if abs(x['MC_Wdecay1_from_t_afterFSR_pdgid']) in had_range and abs(x['MC_Wdecay2_from_t_afterFSR_pdgid']) in had_range else False if abs(x['MC_Wdecay1_from_t_afterFSR_pdgid']) in lep_range and abs(x['MC_Wdecay2_from_t_afterFSR_pdgid']) in lep_range  else 'Error: Boson' if abs(x['MC_Wdecay1_from_t_afterFSR_pdgid']) in bos_range or abs(x['MC_Wdecay2_from_t_afterFSR_pdgid']) in bos_range else 'Error: 0' if abs(x['MC_Wdecay1_from_t_afterFSR_pdgid'])==0 or abs(x['MC_Wdecay2_from_t_afterFSR_pdgid'])==0 else 'Error: Mismatch' ,axis=1)

    # Check that we don't have any funny business going on
    # if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: Boson').any():      # Don't want any bosons from tbar
    #     print("WARNING: Decay 1 and/or 2 from tbar identified as boson.")
    # if (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: Boson').any():         # Don't want any bosons from t
    #     print("WARNING: Decay 1 and/or 2 from t identified as boson.")
    # if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: 0').any():          # Don't want any 0 pdgid from tbar
    #     print("WARNING: Decay 1 and/or 2 from tbar has 0 pdgid.")
    # if (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: 0').any():             # Don't want any 0 pdgid from t
    #     print("WARNING: Decay 1 and/or 2 from t has 0 pdgid.")
    # if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: Mismatch').any():   # Want decay 1 and 2 from tbar to agree
    #     print("WARNING: Decay 1 and 2 from tbar disagree on decay type.")
    # if (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: Mismatch').any():      # Want decay 1 and 2 from t to agree
    #     print("WARNING: Decay 1 and 2 from t disagree on decay type.")
    # if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']==df_truth_W['Wdecay_from_t_isHadronicDecay']).any():   # Want exactly one hadronic and one leptonic decay per event
    #     print("WARNING: Some events have two hadronic or two leptonic decays.")

    # If W from tbar has 0 pdgid, set isHadronicDecay as the opposite of what it is for W from t
    df_truth_W['Wdecay_from_tbar_isHadronicDecay'] = df_truth_W.apply(lambda x : True if x['Wdecay_from_tbar_isHadronicDecay']=='Error: 0' and x['Wdecay_from_t_isHadronicDecay']==False else False if x['Wdecay_from_tbar_isHadronicDecay']=='Error: 0' and x['Wdecay_from_t_isHadronicDecay']==True else x['Wdecay_from_tbar_isHadronicDecay'],axis=1)

    # If W from t has 0 pdgid, set isHadronicDecay as the oppposite of what it is for W from tbar
    df_truth_W['Wdecay_from_t_isHadronicDecay'] = df_truth_W.apply(lambda x : True if x['Wdecay_from_t_isHadronicDecay']=='Error: 0' and x['Wdecay_from_tbar_isHadronicDecay']==False else False if x['Wdecay_from_t_isHadronicDecay']=='Error: 0' and x['Wdecay_from_tbar_isHadronicDecay']==True else x['Wdecay_from_t_isHadronicDecay'],axis=1)

    # One more check for funny business
    # if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']==df_truth_W['Wdecay_from_t_isHadronicDecay']).any():   # Want exactly one hadronic and one leptonic decay per event
    #     print("WARNING: Some events have two hadronic or two leptonic decays.")
    # if (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: Boson').any() or (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: 0').any() or (df_truth_W['Wdecay_from_tbar_isHadronicDecay']=='Error: Mismatch').any():
    #     print("WARNING: W from tbar stil has an error somewhere. This may mess up data sorting.")
    # if (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: Boson').any() or (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: 0').any() or (df_truth_W['Wdecay_from_t_isHadronicDecay']=='Error: Mismatch').any():
    #     print("WARNING: W from t stil has an error somewhere. This may mess up data sorting.")


    print("Hadronic and leptonic W's identified.")

    # Make truth df
    truth_keys = []
    for v in ['pt','eta','phi','m']:
        for p in ['MC_thad_afterFSR_','MC_tlep_afterFSR_']:
            truth_keys.append(p+v)
    df_truth = ak.to_pandas({key.split('_')[1][0:2]+'_'+key.split('_')[-1]:parton_tree[key]/1000 if key.split('_')[-1]=='pt' or key.split('_')[-1]=='m' else parton_tree[key] for key in truth_keys})

    # Get the wh and wl information (and ttbar if desired)
    for v in ['pt','eta','phi','m']:

        # Set wh and wl variables using isHadronicDecay information
        df_truth['wh_'+v] = df_truth_W.apply(lambda x : x['MC_W_from_tbar_afterFSR_'+v] if x['Wdecay_from_tbar_isHadronicDecay']==True else x['MC_W_from_t_afterFSR_'+v],axis=1)    # Sort through W df and find whad
        df_truth['wl_'+v] = df_truth_W.apply(lambda x : x['MC_W_from_t_afterFSR_'+v] if x['Wdecay_from_tbar_isHadronicDecay']==True else x['MC_W_from_tbar_afterFSR_'+v],axis=1)    # Sort through W df and find wlep

        # Include ttbar (if desired)
        if ttbar_addon: 
            df_truth['ttbar_'+v] = parton_tree['MC_ttbar_afterFSR_'+v]/1000 if v=='pt' or v=='m' else parton_tree['MC_ttbar_afterFSR_'+v]

        # Convert from MeV to GeV
        if v=='pt' or v=='m':
            for p in ['wh_','wl_']:
                df_truth[p+v] = df_truth[p+v]/1000

    # Include jet match info (if desired)
    if withJetMatch:
        padded_matches = np.asarray(ak.fill_none(ak.pad_none(reco_tree['jet_isTruth'], jn, clip=True), 0))
        for j in range(jn):
            df_truth['j'+str(j+1)+'_isTruth'] = padded_matches[:,j]
                

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
    root_file = uproot.open('/data/jchishol/mc16e/mntuple_ttbar_'+name+'.root')
    reco_tree = root_file['reco'].arrays()
    parton_tree = root_file['parton'].arrays()

    # Close root file
    root_file.close()

    # Need to make cuts on events if using the non-fixed files (otherwise already done and no need to do it again!)
    if 'fixed+match' not in name:
        # Remove events with nan y values
        for var in ['MC_thad_afterFSR_y','MC_tlep_afterFSR_y','MC_Wdecay1_from_t_afterFSR_y','MC_Wdecay2_from_t_afterFSR_y','MC_Wdecay1_from_tbar_afterFSR_y','MC_Wdecay2_from_tbar_afterFSR_y']:
            sel = np.invert(np.isnan(parton_tree[var]))
            parton_tree = parton_tree[sel]
            reco_tree = reco_tree[sel]

        # Remove isDummy events and !isMatched events
        sel = reco_tree['isMatched']==1
        parton_tree = parton_tree[sel]
        reco_tree = reco_tree[sel]
        sel = parton_tree['isDummy']==0
        parton_tree = parton_tree[sel]
        reco_tree = reco_tree[sel]

    # Remove events with met_met < 20 GeV (to match Tao's cut) (not sure KLFitter does this though)
    sel = reco_tree['met_met']/1000 >= 20
    parton_tree = parton_tree[sel]
    reco_tree = reco_tree[sel]

    # Get the array of jet dataframes
    print('Getting jet dataframes ...')
    df_jets = getJetsData(reco_tree,jn)

    # Get the other reco dataframe
    print('Getting other dataframe ...')
    df_other = getOtherData(reco_tree)

    # Get the truth dataframe
    print('Getting truth dataframe ...')
    df_truth = getTruthData(parton_tree,reco_tree,ttbar_addon,withJetMatch)
    

    return df_jets, df_other, df_truth


# ---------- FUNCTION FOR SAVING H5 FILE ---------- #

def saveH5File(df_jets,df_other,df_truth,save_name):
    """
    Create and save h5 file with the reco and truth variables
    Input: array of jet dataframes, other dataframe, truth dataframe, and save file name (e.g.'1_parton_ejets_addon_ttbar+jetMatch')
    Output: h5 file with jet, other, and truth (th, tl, and possibly ttbar) 4-vectors
    """

    # Creating h5 file for input in the neural network
    f = h5py.File("/data/jchishol/ML_Data/variables_ttbar_"+save_name+".h5","w")  # "w" means initializing file

    # Create datasets for jets
    for j,df in enumerate(df_jets):
        for v in df.columns:
            f.create_dataset('j'+str(j+1)+'_'+v,data=df[v])

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
    Input: the file name types as an array (e.g. ['parton_ejets_addon_ttbar+jetMatch', 'parton_mjets_addon_ttbar+jetMatch'], the file numbers, and save file name
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

    print('Saved: variables_ttbar_'+save_name+'.h5')


# ---------- FUNCTION FOR SAVING XMAXMEAN ---------- #

def saveMaxMean(name):
    """
    Saves numpy array of [max,mean] values for X and Y variables
    Input: file name type
    Output: two numpy files of [max, mean] values for X and Y variables
    """

    print('Opening file ...')

    # Load data
    f = h5py.File('/data/jchishol/ML_Data/variables_ttbar_'+name+'.h5','r')    

    print('File opened.')

    # Create data frame
    df = pd.DataFrame({key: np.array(f.get(key)) for key in list(f.keys())})

    print('Dataframe created')

    # Initialize arrays of maxmean
    X_maxmean=[]
    Y_maxmean=[]

    # Get the number of jets per event
    jn = len(list(filter(lambda a: 'j' in a and 'pt' in a, f.keys())))
    f.close()

    # Jets
    for j in range(jn):

        # Calculate px and py
        df['j'+str(j+1)+'_px'] = df['j'+str(j+1)+'_pt']*np.cos(df['j'+str(j+1)+'_phi'])
        df['j'+str(j+1)+'_py'] = df['j'+str(j+1)+'_pt']*np.sin(df['j'+str(j+1)+'_phi'])
   
        # Append max and mean
        for v in ['pt','px','py','eta','m','isbtag']:
            X_maxmean.append([df['j'+str(j+1)+'_'+v].abs().max(),df['j'+str(j+1)+'_'+v].mean()])
        
        # Also append isTrue if using the jet matching
        if 'jetMatch' in name:
            Y_maxmean.append([df['j'+str(j+1)+'_isTruth'].abs().max(),df['j'+str(j+1)+'_isTruth'].mean()])

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
    particles = ['th_','wh_','tl_','wl_','ttbar_'] if 'addon_ttbar' in name else ['th_','wh_','tl_','wl_']
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
addon_tag = '_addon_ttbar+jetMatch04' if ttbar_addon*withJetMatch else '_addon_ttbar' if ttbar_addon else '_addon_jetMatch04' if withJetMatch else ''

# Part 1: create separate h5 files (will need to change the file number each time this is run)
# file_num = 0
# file_name = '_parton_mjets'
# for file_num in range(8):
#     df_jets, df_other, df_truth = getDataframes(str(file_num)+file_name+'_fixed+match0d4',ttbar_addon,withJetMatch,jn=6)
#     saveH5File(df_jets,df_other,df_truth,str(file_num)+file_name+addon_tag)
#     del df_jets
#     del df_other
#     del df_truth

# Part 2: Put the h5 files together
#combineH5Files(['parton_ejets'+addon_tag,'parton_mjets'+addon_tag],[0,1,2,3,4,6,7],'parton_e+mjets'+addon_tag+'_train')  # Use files 0-4 for training
#combineH5Files(['parton_ejets'+addon_tag],[5],'parton_ejets'+addon_tag+'_test')        # Use file 5 for testing
#combineH5Files(['parton_mjets'+addon_tag],[5],'parton_mjets'+addon_tag+'_test')        # Use file 5 for testing


# Part 3: Create and save numpy array of X and Y max and mean values
saveMaxMean('parton_e+mjets'+addon_tag+'_train')







print ("done :)")
