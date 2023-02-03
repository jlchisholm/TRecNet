# Import useful packages
import uproot
import h5py
import pandas as pd
import awkward as ak
import numpy as np
import vector
import itertools
import tk as tkinter
from Plotter import *


# ---------- FUNCTION DEFINITIONS ---------- #


# Imports data from 0th file and creates a data frame
# Input: reco_method (i.e.'KLFitter' or 'Chi2' or 'PseudoTop') and file name
# Output: data frame
def Create_DF(reco_method,filename): 
	
    # Open first root file and its trees
    file0 = uproot.open('/mnt/xrootdg/jchishol/mntuples_08_01_22/'+filename)


    tree_nom = file0['nominal'].arrays()
    tree_up = file0['CategoryReduction_JET_Pileup_RhoTopology__1up'].arrays()
    tree_down = file0['CategoryReduction_JET_Pileup_RhoTopology__1down'].arrays()
    keys = file0['nominal'].keys()

    # Did an met_met cut for the ML data, so let's match it here too
    #sel = tree_nom['met_met']/1000 >= 20
    #tree_nom = tree_nom[sel]

    # Also split the datafile the same way we did when making the train/test h5 files
    #split_point = int(np.round(len(tree_nom['eventNumber'])*0.85))
    #tree_nom = tree_nom[split_point:]

    # Only take events with the same event numbers as the test data (do this in the main method)
    #sel = np.isin(tree_nom['eventNumber'],eventnumbers)
    #tree_nom = tree_nom[sel]

    # Initialize list of variable names
    reco_names = []
    truth_names = []
    model_name = 'PseudoTop_Reco_' if reco_method=='PseudoTop' else 'TtresChi2_' if reco_method=='Chi2' else 'klfitter_bestPerm_'


    # Add reco variables to be plotted
    for par in [top_had,top_lep,top_antitop]:      # For each particle

        # Get all the possible names for this particle
        alt_par_names = list(par.alt_names)
        alt_par_names.append(par.name)

        for var in par.observables:                  # For each variable of that particle

            # Reset
            found_reco=False
            found_truth=False

            # Get all the possible names for this variable
            alt_var_names = list(var.alt_names)
            alt_var_names.append(var.name)

            for (par_name,var_name) in list(itertools.product(alt_par_names,alt_var_names)):
                if model_name+par_name+'_'+var_name in keys:
                    reco_names.append(['reco_'+par.name+'_'+var.name,model_name+par_name+'_'+var_name])
                    found_reco=True
                if model_name+var_name in keys: # (particularly, chi_tt)
                    reco_names.append(['reco_'+par.name+'_'+var.name,model_name+var_name])
                    found_reco=True
                if 'MC_'+par_name+'_afterFSR_'+var_name in keys:
                    truth_names.append(['truth_'+par.name+'_'+var.name,'MC_'+par_name+'_afterFSR_'+var_name])
                    found_truth=True
                if 'MC_'+var_name in keys:  # (particularly, chi_tt)
                    truth_names.append(['truth_'+par.name+'_'+var.name,'MC_'+var_name])
                    found_truth = True
                # if 'MC_'+par_name+var_name in keys: # (in particular, for Pout)
                #     df['truth_'+par.name+'_'+var.name] = ak.to_pandas(tree_nom['MC_'+par_name+var_name])    # Append the true values
                #     found_truth=True

                # Stop looking for the correct names once we find them
                if found_reco and found_truth:
                    break

    # Create dataframe(s)!
    names = truth_names+reco_names
    df_nom = ak.to_pandas({new_name:tree_nom[old_name] for (new_name,old_name) in names})
    df_up = ak.to_pandas({'sysUP_'+new_name.lstrip('reco_'):tree_up[old_name] for (new_name,old_name) in reco_names})
    df_down = ak.to_pandas({'sysDOWN_'+new_name.lstrip('reco_'):tree_down[old_name] for (new_name,old_name) in reco_names})

    df_nom['eventNumber'] = tree_nom['eventNumber']
    df_up['eventNumber'] = tree_up['eventNumber']
    df_down['eventNumber'] = tree_down['eventNumber']

    # Include number of jets, so we can look at how they might compare
    df_nom['jet_n'] = tree_nom['jet_n']
    df_up['jet_n'] = tree_up['jet_n']
    df_down['jet_n'] = tree_down['jet_n']

    # Bring in Chi2 for ... Chi2 lol
    if reco_method=='Chi2':
        df_nom['chi2'] = tree_nom['TtresChi2_Chi2']
        df_up['chi2'] = tree_up['TtresChi2_Chi2']
        df_down['chi2'] = tree_down['TtresChi2_Chi2']

        # Cut on eta (had some weird really really large numbers that seem to be reconstruction fails)
        df_nom = df_nom[df_nom['reco_ttbar_eta']<1000]
        df_nom = df_nom[df_nom['reco_ttbar_eta']>-1000]
        df_up = df_up[df_up['sysUP_ttbar_eta']<1000]
        df_up = df_up[df_up['sysUP_ttbar_eta']>-1000]
        df_down = df_down[df_down['sysDOWN_ttbar_eta']<1000]
        df_down = df_down[df_down['sysDOWN_ttbar_eta']>-1000]


    elif reco_method=='KLFitter4' or reco_method=='KLFitter6':

        # Note: some events did not have a logLikelihood calculated -- we will need to pad these events to make sure nothing gets shifted weird, and then cut them
        df_nom['logLikelihood'] = ak.flatten(ak.fill_none(ak.pad_none(tree_nom['klfitter_logLikelihood'],1),np.nan))
        df_up['logLikelihood'] = ak.flatten(ak.fill_none(ak.pad_none(tree_up['klfitter_logLikelihood'],1),np.nan))
        df_down['logLikelihood'] = ak.flatten(ak.fill_none(ak.pad_none(tree_down['klfitter_logLikelihood'],1),np.nan))


    # Drop bad klfitter events
    for df in [df_nom, df_up, df_down]:
        df.replace([np.inf,-np.inf],np.nan,inplace=True)
        df.dropna(inplace=True)


    # Close root file
    file0.close()

    print('Appended file '+filename+' to '+reco_method+' dataframe.')	

    # Return main data frame
    return df_nom, df_up, df_down


# Appends data from the rest of the files to the existing data frame
# Note: currently used for KLFitter, not ML (only have one result file from the testing.ipynb)
# Input: data frame and name of file to be appended
# Output: data frame
def Append_Data(df_nom,df_up,df_down,reco_method,filename):

    # Get the data from the new file
    df_nom_addon, df_up_addon, df_down_addon = Create_DF(reco_method,filename)
        
    # Append data to the main data frame
    df_nom = pd.concat([df_nom,df_nom_addon],axis=0,ignore_index=True)
    df_up = pd.concat([df_up,df_up_addon],axis=0,ignore_index=True)
    df_down = pd.concat([df_down,df_down_addon],axis=0,ignore_index=True)

    #print('Appended file '+filename)

    return df_nom,df_up,df_down






### ----- MAIN CODE ----- ###


# Define all the variables and their ranges we want
scale = '[GeV]'
pt = Observable('pt', 'p_T', np.arange(0,550,50), unit=scale)
px = Observable('px','p_x',np.arange(-250,250,50),unit=scale)
py = Observable('py','p_y',np.arange(-250,250,50),unit=scale)
eta = Observable('eta', '\eta', np.arange(-6,7.5,1.5))
y = Observable('y', 'y', np.arange(-2.5,3,0.5))
phi = Observable('phi', '\phi', np.arange(-3,3.75,0.75))
m_t = Observable('m','m', np.arange(100,260,20),unit=scale)
E_t = Observable('E','E', np.arange(100,1600,150),unit=scale)
#pout_t = Observable('pout','p_{out}',range(-275,325,50), unit=scale,alt_names=['Pout'])
m_tt = Observable('m','m', np.arange(200,1700,150),unit=scale)
E_tt = Observable('E','E', np.arange(200,2700,250),unit=scale)
dphi_tt = Observable('dphi','|\Delta\phi|', np.arange(0,3.6,0.45),alt_names=['deltaPhi'])
deta_tt = Observable('deta','|\Delta\eta|', np.arange(0,10,1),alt_names=['deltaEta'])
Ht_tt = Observable('Ht','H_T',np.arange(0,1200,120),unit=scale,alt_names=['HT'])
yboost_tt = Observable('yboost','y_{boost}',np.arange(-3,3.75,0.75),alt_names=['y_boost'])
#ystar_tt = Observable('ystar','y_{star}',np.arange(-2.5,3,0.5),alt_names=['y_star'])
chi_tt = Observable('chi','\chi',np.arange(0,25,2.5),alt_names=['chi_tt'])

# Define the particles we want
top_had = Particle('th','t,had',[pt,px,py,eta,y,phi,m_t,E_t],alt_names=['thad','topHad','top_had'])
top_lep = Particle('tl','t,lep',[pt,px,py,eta,y,phi,m_t,E_t],alt_names=['tlep','topLep','top_lep'])
top_antitop = Particle('ttbar','t\overline{t}',[pt,px,py,eta,y,phi,m_tt,E_tt,dphi_tt,deta_tt,Ht_tt,yboost_tt,chi_tt])




# Create the data frames and dataset objects for KLFitter, PseudoTop, and Chi2
#KLF4_df, KLF4_sysUP_df, KLF4_sysDOWN_df = Create_DF('KLFitter4','mc16a_4j/mntuple_ljets_1_jetMatch04.root')
KLF6_df, KLF6_sysUP_df, KLF6_sysDOWN_df = Create_DF('KLFitter6','mc16a_6j/mntuple_ljets_1_jetMatch04.root')
PT_df, PT_sysUP_df, PT_sysDOWN_df = Create_DF('PseudoTop','mc16a_6j/mntuple_ljets_1_jetMatch04.root')
Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df = Create_DF('Chi2','mc16a_6j/mntuple_ljets_1_jetMatch04.root')


for n in range(1,100):
    #if n in range(2,12):  # Use the same files as were used for testing!
        #KLF4_df, KLF4_sysUP_df, KLF4_sysDOWN_df = Append_Data(KLF4_df, KLF4_sysUP_df, KLF4_sysDOWN_df,'KLFitter4','mc16a_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        #PT_df, PT_sysUP_df, PT_sysDOWN_df = Append_Data(PT_df, PT_sysUP_df, PT_sysDOWN_df,'PseudoTop','mc16a_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        #Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df = Append_Data(Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df,'Chi2','mc16a_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')

    if n in range(2,23):
        KLF6_df, KLF6_sysUP_df, KLF6_sysDOWN_df= Append_Data(KLF6_df, KLF6_sysUP_df, KLF6_sysDOWN_df,'KLFitter6','mc16a_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        PT_df, PT_sysUP_df, PT_sysDOWN_df = Append_Data(PT_df, PT_sysUP_df, PT_sysDOWN_df,'PseudoTop','mc16a_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df = Append_Data(Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df,'Chi2','mc16a_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')

    #if n in range(1,35) and n!=14 and n!=15:  # Use the same files as were used for testing!
        #KLF4_df, KLF4_sysUP_df, KLF4_sysDOWN_df = Append_Data(KLF4_df, KLF4_sysUP_df, KLF4_sysDOWN_df,'KLFitter4','mc16d_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        #PT_df, PT_sysUP_df, PT_sysDOWN_df = Append_Data(PT_df, PT_sysUP_df, PT_sysDOWN_df,'PseudoTop','mc16d_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        #Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df = Append_Data(Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df,'Chi2','mc16d_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')

    if n in [1,2,3,6,10,11,13,14,15,16,17,18,37,38,58,60,64,66,68] or n in range(72,83):
        KLF6_df, KLF6_sysUP_df, KLF6_sysDOWN_df = Append_Data(KLF6_df, KLF6_sysUP_df, KLF6_sysDOWN_df,'KLFitter6','mc16d_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        PT_df, PT_sysUP_df, PT_sysDOWN_df = Append_Data(PT_df, PT_sysUP_df, PT_sysDOWN_df,'PseudoTop','mc16d_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df = Append_Data(Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df,'Chi2','mc16d_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')

    #if n in range(1,36):  # Use the same files as were used for testing!
        #KLF4_df, KLF4_sysUP_df, KLF4_sysDOWN_df = Append_Data(KLF4_df, KLF4_sysUP_df, KLF4_sysDOWN_df,'KLFitter4','mc16e_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        #PT_df, PT_sysUP_df, PT_sysDOWN_df = Append_Data(PT_df, PT_sysUP_df, PT_sysDOWN_df,'PseudoTop','mc16e_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        #Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df = Append_Data(Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df,'Chi2','mc16e_4j/mntuple_ljets_'+str(n)+'_jetMatch04.root')

    if n in range(1,31) and n!=7 and n!=23:
        KLF6_df, KLF6_sysUP_df, KLF6_sysDOWN_df = Append_Data(KLF6_df, KLF6_sysUP_df, KLF6_sysDOWN_df,'KLFitter6','mc16e_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        PT_df, PT_sysUP_df, PT_sysDOWN_df = Append_Data(PT_df, PT_sysUP_df, PT_sysDOWN_df,'PseudoTop','mc16e_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')
        Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df = Append_Data(Chi2_df, Chi2_sysUP_df, Chi2_sysDOWN_df,'Chi2','mc16e_6j/mntuple_ljets_'+str(n)+'_jetMatch04.root')


# f_KLF4 = uproot.recreate('/mnt/xrootdg/jchishol/mntuples_08_01_22/KLF4_Results.root')
# KLF4_truth_df = KLF_df.iloc[:,KLF4_df.columns.str.startswith("truth")]
# KLF4_truth_df['eventNumber'] = KLF4_df['eventNumber']
# KLF4_reco_df = KLF4_df.iloc[:,KLF4_df.columns.str.startswith("reco")]
# KLF4_sysUP_df = KLF4_df.iloc[:,KLF4_df.columns.str.startswith("sysUP")]
# KLF4_sysDOWN_df = KLF4_df.iloc[:,KLF4_df.columns.str.startswith("sysDOWN")]
# f_KLF4["truth"] = KLF4_truth_df
# f_KLF4["reco"] = KLF4_reco_df
# f_KLF4["sysUP"] = KLF4_sysUP_df
# f_KLF4["sysDOWN"] = KLF4_sysDOWN_df



f_KLF6 = uproot.recreate('/mnt/xrootdg/jchishol/mntuples_08_01_22/KLF6_Results.root')
KLF6_truth_df = KLF6_df.iloc[:,KLF6_df.columns.str.startswith("truth")]
KLF6_truth_df['eventNumber'] = KLF6_df['eventNumber']
KLF6_reco_df = KLF6_df.iloc[:,KLF6_df.columns.str.startswith("reco")]
KLF6_reco_df['logLikelihood'] = KLF6_df['logLikelihood']
#KLF6_sysUP_df = KLF6_df.iloc[:,KLF6_df.columns.str.startswith("sysUP")]
#KLF6_sysDOWN_df = KLF6_df.iloc[:,KLF6_df.columns.str.startswith("sysDOWN")]
f_KLF6["truth"] = KLF6_truth_df
f_KLF6["reco"] = KLF6_reco_df
f_KLF6["sysUP"] = KLF6_sysUP_df
f_KLF6["sysDOWN"] = KLF6_sysDOWN_df

f_PT = uproot.recreate('/mnt/xrootdg/jchishol/mntuples_08_01_22/PT_Results.root')
PT_truth_df = PT_df.iloc[:,PT_df.columns.str.startswith("truth")]
PT_truth_df['eventNumber'] = PT_df['eventNumber']
PT_reco_df = PT_df.iloc[:,PT_df.columns.str.startswith("reco")]
#PT_sysUP_df = PT_df.iloc[:,PT_df.columns.str.startswith("sysUP")]
#PT_sysDOWN_df = PT_df.iloc[:,PT_df.columns.str.startswith("sysDOWN")]
f_PT["truth"] = PT_truth_df
f_PT["reco"] = PT_reco_df
f_PT["sysUP"] = PT_sysUP_df
f_PT["sysDOWN"] = PT_sysDOWN_df

f_Chi2 = uproot.recreate('/mnt/xrootdg/jchishol/mntuples_08_01_22/Chi2_Results.root')
Chi2_truth_df = Chi2_df.iloc[:,Chi2_df.columns.str.startswith("truth")]
Chi2_truth_df['eventNumber'] = Chi2_df['eventNumber']
Chi2_reco_df = Chi2_df.iloc[:,Chi2_df.columns.str.startswith("reco")]
Chi2_reco_df['chi2'] = Chi2_df['chi2']
#Chi2_sysUP_df = Chi2_df.iloc[:,Chi2_df.columns.str.startswith("sysUP")]
#Chi2_sysDOWN_df = Chi2_df.iloc[:,Chi2_df.columns.str.startswith("sysDOWN")]
f_Chi2["truth"] = Chi2_truth_df
f_Chi2["reco"] = Chi2_reco_df
f_Chi2["sysUP"] = Chi2_sysUP_df
f_Chi2["sysDOWN"] = Chi2_sysDOWN_df

print('Done! :)')
    
