import uproot
import h5py
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib
#matplotlib.use('GTK3Agg')   # need this one if you want plots displayed
import vector
import os
from __main__ import *

# Define some helpful ranges
had_range = list(range(1,9))+list(range(-8,0))                     # Quarks have pdgid 1 to 8 (negatives indicate antiparticles)
lep_range = list(range(11,19))+list(range(-18,10))                 # Leptons have pdgid 11 to 18
bos_range = list(range(21,26))+list(range(-25,20))+[9,37,-9,-37]   # Bosons have pdgid 21 to 25, and 9, and 37
light_quarks = list(range(-4,0)) + list(range(1,5))
b_quarks = [-5,5]



class jetMatcher:
    """ 
    A class for matching ttbar decay products to reco level jets.

        Methods:
            pruneTrees: Removes events we can't use from the trees (i.e. events with nan y and non-semileptonic events)
            getMatches: Matches ttbar decay products to reco-level jets.
            addMatchesToFile: Gets the jet match tags and creates a new root file from the old one, including these new tags.

    """

    def __init__(self):
        print("Creating jetMatcher.")

    def pruneTree(self,nom_tree):
        """ 
        Removes events we can't use from the trees (i.e. events with nan y and non-semileptonic events)

            Parameters:
                nom_tree (root tree): Nominal tree from the root file.

            Returns:
                nom_tree (root tree): Nominal tree, with bad events removed.
        """

        # Don't want nan (or ridiculously negative) values
        #for var in ['MC_thad_afterFSR_y','MC_tlep_afterFSR_y','MC_W_from_thad_y','MC_W_from_tlep_y']:
        #    sel = np.invert(np.isnan(nom_tree[var]))
        
        #for var in ['MC_thad_afterFSR_eta','MC_tlep_afterFSR_eta','MC_W_from_thad_eta','MC_W_from_tlep_eta']:
        #    sel = sel * (nom_tree[var]>=-100)
        sel = nom_tree['MC_ttbar_afterFSR_eta']>-100

        # Only want semi-leptonic events
        sel = sel*(nom_tree['semileptonicEvent']==1)

        # Make these selections
        nom_tree = nom_tree[sel]

        return nom_tree


    def getMatches(self,nom_tree, dR_cut=1.0, allowDoubleMatch=True):
        """
        Matches ttbar decay products to reco-level jets.

            Parameters:
                nom_tree (root tree): Nominal tree from the root file.
            
            Options:
                dR_cut (float): A threshold which the dR for all matches must be below (default: 1.0).
                allowDoubleMatch (bool): Whether or not two or more decay products are allowed to be matched to the same jet (default: True).

            Returns:
                isttbarJet (jagged array of bools): Tags for each jet in each event, where 0 means it was not matched to something, and 1 means it was.
                match_info (ndarray): Array of match info for each decay product in all events (form: [event index, decay particle, matched jet, (absolute) jet pdgid, dR for the match, fractional delta pt for the match]). 
        """

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


            # Print every once in a while so we know there's progress
            if (i+1)%100000==0 or i==num_events-1:
                print ('Jet Events Processed: '+str(i+1))


        return isttbarJet, match_info


    def addMatchesToFile(self,file,dR_cut=1.0,allowDoubleMatching=True):
        """
        Gets the jet match tags and creates a new root file from the old one, including these new tags.

            Parameters:
                file (root file): Root file you'd like to add the jet matches to.

            Options:
                dR_cut (float): A threshold which the dR for all matches must be below (default: 1.0).
                allowDoubleMatch (bool): Whether or not two or more decay products are allowed to be matched to the same jet (default: True).

            Returns:
                Creates a new root file that includes the systematic uncertainty trees, as well as the nominal tree with the new jet match tags included as 'jet_isTruth'.
        """

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
        nom_tree = self.pruneTree(nom_tree)

        # Get the jet matches
        print('Matching jets ...')
        isttbarJet, matching_info = self.getMatches(nom_tree, dR_cut, allowDoubleMatching)

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



class filePrep:
    """ 
    A class for creating all the necessary files for training or testing the machine learning model.

        Methods:
            getJetsData: Creates an array of dataframes for the jets, including pt, eta, phi, m, and btag for each jet.
            getOtherData: Creates a dataframe for the other (lep, met) variables. Includes lep_pt, lep_eta, lep_phi, met_met, and met_phi.
            getTruthData: Creates a dataframe for the truth variables, including: pt, eta, phi, m, y, E, and pout for thad and tlep; and pt, eta, phi, m, y, E, dphi, Ht, chi, and yboost for ttbar; pt, eta, phi, m for whad and wlep; and isTruth for each of the jets.
            getDataframes: Creates the jet, other, and truth dataframes from a specified file.
            makeH5File: Creates and saves h5 file with the reco and truth variables.
            combineH5Files: Combines several h5 files into one file.
            saveMaxMean: Saves numpy array of [max,mean] values for X (reco) and Y (truth) variables.
            

    """

    def __init__(self):
        print("Creating filePrepper.")


    def getJetsData(self,reco_tree,jn=6):
        """
        Creates an array of dataframes for the jets, including pt, eta, phi, m, and btag for each jet.

            Parameters:
                reco_tree (root tree): Reco level tree from ROOT file

            Options:
                jn (int): Number of jets you want per event, with padding where need be (default: 6)

            Returns:
                df_jets (array of dataframes): An array of <jn> dataframes (one for each of the jets), containing jet data.
        """

        # Create an array of <jn> jet dataframes
        df_jets = [pd.DataFrame() for _ in range(jn)]
        
        # Pad pt, eta, phi, e with 0's for events with less than <jn> jets, and add these variables to each of the jet dataframes
        for v in ['pt','eta','phi','e']:
            padded_vars = np.asarray(ak.fill_none(ak.pad_none(reco_tree['jet_'+v], jn, clip=True), 0.))
            for j, df in enumerate(df_jets):
                df[v] = padded_vars[:,j]

        # Pad btag with -1.'s
        #padded_btags = np.asarray(ak.fill_none(ak.pad_none(reco_tree['jet_btagged'], jn, clip=True), -1))
        padded_btags = np.asarray(ak.fill_none(ak.pad_none(reco_tree['jet_isbtagged_DL1r_70'], jn, clip=True), -1))

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


    def getOtherData(self,reco_tree):
        """
        Creates a dataframe for the other (lep, met) variables. Includes lep_pt, lep_eta, lep_phi, met_met, and met_phi.

            Parameters:
                reco_tree (root tree): Reco level tree from ROOT file
            
            Returns:
                df_other (dataframe): Dataframe for the other (lep, met) variables.

        """

        # Create dataframe for other variables (note divide lep_pt and met_met by 1000 to convert from MeV to GeV)
        # NOTE: also including jet_n so we can later make a cut on these perhaps?
        keys = ['lep_pt','lep_eta','lep_phi','met_met','met_phi','jet_n']
        df_other = ak.to_pandas({key:reco_tree[key]/1000 if key=='lep_pt' or key=='met_met' else reco_tree[key] for key in keys})

        print('Other dataframe created.')

        return df_other


    def getTruthData(self,nom_tree,jn=6):
        """
        Creates a dataframe for the truth variables, including: pt, eta, phi, m, y, E, and pout for thad and tlep; and pt, eta, phi, m, y, E, dphi, Ht, chi, and yboost for ttbar; pt, eta, phi, m for whad and wlep; and isTruth for each of the jets.
        
            Parameters: 
                nom_tree (root tree): Parton level tree from ROOT file.
            
            Options:
                jn (int): Number of jets you want per event, with padding where need be (default: 6).
            
            Returns:
                df_truth (dataframe): Dataframe for the truth variables.
        """

        # Make truth df (starting with thad and tlep variables)
        truth_keys = []
        for v in ['pt','eta','phi','m','y','E']:
            for p in ['MC_thad_afterFSR_','MC_tlep_afterFSR_']:
                truth_keys.append(p+v)
        df_truth = ak.to_pandas({key.split('_')[1][0:2]+'_'+key.split('_')[-1]:nom_tree[key] for key in truth_keys})
        df_truth['th_pout'] = nom_tree['MC_thad_Pout']
        df_truth['tl_pout'] = nom_tree['MC_tlep_Pout']

        # Include ttbar
        for v in ['pt','eta','phi','m','y','E']:
            df_truth['ttbar_'+v] = nom_tree['MC_ttbar_afterFSR_'+v]
        df_truth['ttbar_dphi'] = nom_tree['MC_deltaPhi_tt']
        df_truth['ttbar_Ht'] = nom_tree['MC_HT_tt']
        df_truth['ttbar_chi'] = nom_tree['MC_chi_tt']
        df_truth['ttbar_yboost'] = nom_tree['MC_y_boost']


        # Get the wh and wl information
        for v in ['pt','eta','phi','m']:

            # Set wh and wl variables using isHadronicDecay information
            df_truth['wh_'+v] = nom_tree['MC_W_from_thad_'+v] 
            df_truth['wl_'+v] = nom_tree['MC_W_from_tlep_'+v] 

        # Include jet match info (if desired)
        padded_matches = np.asarray(ak.fill_none(ak.pad_none(nom_tree['jet_isTruth'], jn, clip=True), 0))
        for j in range(jn):
            df_truth['j'+str(j+1)+'_isTruth'] = padded_matches[:,j]

        # Convert from MeV to GeV
        for col in df_truth:
            if col.split('_')[1] in ['pt','m','E','pout','Ht']:
                df_truth[col] = df_truth[col]/1000


        print('Truth dataframe created.')

        return df_truth


    def getDataframes(self,root_file,jn=6,met_cut=0):
        """
        Creates the jet, other, and truth dataframes from a specified file.

            Parameters: 
                root_file (root file): Root file you'd like to extract data from.

            Options:
                jn (int): Number of jets you want per event, with padding where need be (default: 6).
                met_cut (int or float): Minimum cut on met_met values (default: 0).
        
            Returns: 
                df_jets (array of dataframes): An array of <jn> dataframes (one for each of the jets), containing jet data.
                df_other (dataframe): Dataframe for the other (lep, met) variables.
                df_truth (dataframe): Dataframe for the truth variables.
        """

        # Import the root file data
        root_file = uproot.open(root_file.path)
        nom_tree = root_file['nominal'].arrays()

        # Close root file
        root_file.close()

        # Remove events with met_met < 20 GeV (to match Tao's cut)
        sel = nom_tree['met_met']/1000 >= met_cut
        nom_tree = nom_tree[sel]

        # Get the array of jet dataframes
        print('Getting jet dataframes ...')
        df_jets = self.getJetsData(nom_tree,jn)

        # Get the other reco dataframe
        print('Getting other dataframe ...')
        df_other = self.getOtherData(nom_tree)

        # Get the truth dataframe
        print('Getting truth dataframe ...')
        df_truth = self.getTruthData(nom_tree,jn)
        

        return df_jets, df_other, df_truth


    def makeH5File(self,file,save_name,jn=6,met_cut=0,save_loc='./'):
        """
        Creates and saves h5 file with the reco and truth variables.

            Parameters: 
                file (root file): Root file you'd like to extract data from.
                save_name (str): Name the file will be saved as.
                
            Options:
                save_loc (str): Location the file will be saved in (default: current directory).
                jn (int): Number of jets you want per event, with padding where need be (default: 6).
                met_cut (int or float): Minimum cut on met_met values (default: 0).

            Returns: 
                Saves an h5 file with jet, other, and truth data. See getJetData, getOtherData, and getTruthData for details on what's included.
        """

        # Get the data to be saved
        df_jets, df_other, df_truth = self.getDataframes(file,jn=jn,met_cut=met_cut)

        # Creating h5 file for input in the neural network
        f = h5py.File(save_loc+save_name+".h5","w")  # "w" means initializing file

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


        print('Saved: '+save_loc+save_name+'.h5')


    def combineH5Files(self,file_list,save_name):
        """
        Combines several h5 files into one file.

            Parameters: 
                file_list (array of files): All the h5 files you want to combine.
                save_name (str): Name the file will be saved as.

            Returns: 
                Saves the combined h5 file (in the same location as the last h5 file in the list).
        """

        # Create a file to combine the data in
        with h5py.File('/data/jchishol/ML_Data/'+save_name+'.h5','w') as h5fw:
            
            current_row = 0   # Keeps track of how many rows of data we have written
            total_len = 0     # Keeps track of how much data we have read
            
            # For each file of that type
            for file in file_list:
                
                # Read the file
                h5fr = h5py.File(file.path,'r')    # Read the file
                
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

                print(file.name+' appended.')

        print('Saved: '+file.path.split(file.name)[0]+save_name+'.h5')


    def saveMaxMean(self,name):
        """
        Saves dictionary of [max,mean] values for X (reco) and Y (truth) variables.
        
            Parameters: 
                name (str): Name of the h5 file you want to calculate the max mean values for.

            Returns: 
                Saves two numpy files of [max, mean] values; one for X (reco) variables and one for Y (truth) variables
        """

        print('Opening file ...')

        # Load data
        f = h5py.File('/data/jchishol/ML_Data/'+name+'.h5','r')    

        print('File opened.')

        # Create data frame
        df = pd.DataFrame({key: np.array(f.get(key)) for key in list(f.keys())})

        # Initialize dictionaries of maxmean
        X_maxmean = {}
        Y_maxmean = {}

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
                X_maxmean['j'+str(j+1)+'_'+v] = [df['j'+str(j+1)+'_'+v].abs().max(),df['j'+str(j+1)+'_'+v].mean()]
            
            # Also append isTrue if using the jet matching
            if 'jetMatch' in name:
                Y_maxmean['j'+str(j+1)+'_isTruth'] = [df['j'+str(j+1)+'_isTruth'].abs().max(),df['j'+str(j+1)+'_isTruth'].mean()]

        print('Jets done.')

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
            X_maxmean[v] = [df[v].abs().max(),df[v].mean()]

        print('Other done')

        # Save array of X maxmean values
        np.save('X_maxmean_'+name,X_maxmean)
        print('Saved: X_maxmean_'+name+'.npy')

        # Calculate px and py for truth
        particles = ['th_','wh_','tl_','wl_','ttbar_']
        for p in particles:
            df[p+'px'] = df[p+'pt']*np.cos(df[p+'phi'])
            df[p+'py'] = df[p+'pt']*np.sin(df[p+'phi'])

            # Append maxmean for all truth variables
            for v in ['pt','px','py','eta','m']:
                Y_maxmean[p+v] = [df[p+v].abs().max(),df[p+v].mean()]

                print('Appended '+p+v)

        # Save Y maxmean arrays
        np.save('Y_maxmean_'+name,Y_maxmean)
        print('Saved: Y_maxmean_'+name+'.npy')