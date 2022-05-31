# Import useful packages
import uproot
import pandas as pd
import awkward as ak
import numpy as np
import vector
import tk as tkinter
import matplotlib
#matplotlib.use('Agg')  # need for not displaying plots when running batch jobs
#matplotlib.use('GTK3Agg')   # need this one if you want plots displayed
from matplotlib import pyplot as plt
from matplotlib import colors
import Plotter


# List some useful strings, organized by particle
level = {'th' : ['klfitter_bestPerm_topHad_','MC_thad_afterFSR_'],
       'tl' : ['klfitter_bestPerm_topLep_','MC_tlep_afterFSR_'],
       'ttbar': ['klfitter_bestPerm_ttbar_','MC_ttbar_afterFSR_']}


# Defines bins/ticks for each variable

# eta
ticks_eta = np.arange(-6,7.5,1.5)
ticks_labels_eta = [str(x) for x in ticks_eta]
ticks_labels_eta[0] = '-'+r'$\infty$'
ticks_labels_eta[-1] = r'$\infty$'

# y
ticks_y = np.arange(-2.5,3,0.5)
ticks_labels_y = [str(x) for x in ticks_y]
ticks_labels_y[0] = '-'+r'$\infty$'
ticks_labels_y[-1] = r'$\infty$'

# phi
ticks_phi = np.arange(-3,3.75,0.75)
ticks_labels_phi = [str(x) for x in ticks_phi]
ticks_labels_phi[0] = '-3.14'
ticks_labels_phi[-1] = '3.14'

# thad or tlep:

# pt
ticks_pt_t = range(0,500,50)
ticks_labels_pt_t = [str(x) for x in ticks_pt_t]
ticks_labels_pt_t[-1] = r'$\infty$'

# m
ticks_m_t = range(100,260,20)
ticks_labels_m_t = [str(x) for x in ticks_m_t]
ticks_labels_m_t[0] = '-'+r'$\infty$'
ticks_labels_m_t[-1] = r'$\infty$'

# E
ticks_E_t = range(100,1000,100)
ticks_labels_E_t = [str(x) for x in ticks_E_t]
ticks_labels_E_t[-1] = r'$\infty$'

# pout
ticks_pout_t = range(-275,325,50)
ticks_labels_pout_t = [str(x) for x in ticks_pout_t]
ticks_labels_pout_t[0] = '-'+r'$\infty$'
ticks_labels_pout_t[-1] = r'$\infty$'

# ttbar:

# pt
ticks_pt_ttbar = range(0,500,50)
ticks_labels_pt_ttbar = [str(x) for x in ticks_pt_ttbar]
ticks_labels_pt_ttbar[-1] = r'$\infty$'

# m
ticks_m_ttbar = range(200,1100,100)
ticks_labels_m_ttbar = [str(x) for x in ticks_m_ttbar]
ticks_labels_m_ttbar[-1] = r'$\infty$'

# E
ticks_E_ttbar = range(200,2200,200)
ticks_labels_E_ttbar = [str(x) for x in ticks_E_ttbar]
ticks_labels_E_ttbar[-1] = r'$\infty$'

# dphi
ticks_dphi_ttbar = np.arange(0,4,0.5)
ticks_labels_dphi_ttbar = [str(x) for x in ticks_dphi_ttbar]
ticks_labels_dphi_ttbar[-1] = r'$\infty$'

# Ht
ticks_Ht_ttbar = range(0,1100,100)
ticks_labels_Ht_ttbar = [str(x) for x in ticks_Ht_ttbar]
ticks_labels_Ht_ttbar[-1] = r'$\infty$'

# yboost
ticks_yboost_ttbar = np.arange(-3,3.5,0.75)
ticks_labels_yboost_ttbar = [str(x) for x in ticks_yboost_ttbar]
ticks_labels_yboost_ttbar[0] = '-'+r'$\infty$'
ticks_labels_yboost_ttbar[-1] = r'$\infty$'

# ystar
ticks_ystar_ttbar = np.arange(-2.5,3,0.5)
ticks_labels_ystar_ttbar = [str(x) for x in ticks_ystar_ttbar]
ticks_labels_ystar_ttbar[0] = '-'+r'$\infty$'
ticks_labels_ystar_ttbar[-1] = r'$\infty$'

# chi
ticks_chi_ttbar = np.arange(0,25,2.5)
ticks_labels_chi_ttbar = [str(x) for x in ticks_chi_ttbar]
ticks_labels_chi_ttbar[-1] = r'$\infty$'




# Define the particles we'll be working with

top_had = Plotter.Particle('th',
                           ['pt','eta','y','phi','m','E','pout'],
                           ['$p_T^{t,had}$','$\eta^{t,had}$','$y^{t,had}$','$\phi^{t,had}$','$m^{t,had}$','$E^{t,had}$','$p_{out}^{t,had}$'],
                           [' [GeV]','','','',' [GeV]',' [GeV]',' [GeV]'],
                           [ticks_pt_t, ticks_eta, ticks_y, ticks_phi, ticks_m_t, ticks_E_t, ticks_pout_t],
                           [ticks_labels_pt_t, ticks_labels_eta, ticks_labels_y, ticks_labels_phi, ticks_labels_m_t, ticks_labels_E_t, ticks_labels_pout_t])

top_lep = Plotter.Particle('tl',
                           ['pt','eta','y','phi','m','E','pout'],
                           ['$p_T^{t,lep}$','$\eta^{t,lep}$','$y^{t,lep}$','$\phi^{t,lep}$','$m^{t,lep}$','$E^{t,lep}$','$p_{out}^{t,lep}$'],
                           [' [GeV]','','','',' [GeV]',' [GeV]',' [GeV]'],
                           [ticks_pt_t, ticks_eta, ticks_y, ticks_phi, ticks_m_t, ticks_E_t, ticks_pout_t],
                           [ticks_labels_pt_t, ticks_labels_eta, ticks_labels_y, ticks_labels_phi, ticks_labels_m_t, ticks_labels_E_t, ticks_labels_pout_t])

top_antitop = Plotter.Particle('ttbar',
                               ['pt','eta','y','phi','m','E','dphi','Ht','yboost','ystar','chi'],
                               ['$p_T^{t\overline{t}}$','$\eta^{t\overline{t}}$','$y^{t\overline{t}}$','$\phi^{t\overline{t}}$','$m^{t\overline{t}}$','$E^{t\overline{t}}$','$\Delta\phi^{t\overline{t}}$','$H_T^{t\overline{t}}$','$y_{boost}^{t\overline{t}}$','$y^{* t\overline{t}}$','$\chi^{t\overline{t}}$'],
                               [' [GeV]','','','',' [GeV]',' [GeV]','',' [GeV]','','',''],
                               [ticks_pt_ttbar, ticks_eta, ticks_y, ticks_phi, ticks_m_ttbar, ticks_E_ttbar, ticks_dphi_ttbar, ticks_Ht_ttbar, ticks_yboost_ttbar, ticks_ystar_ttbar,ticks_chi_ttbar],
                               [ticks_labels_pt_ttbar, ticks_labels_eta, ticks_labels_y, ticks_labels_phi, ticks_labels_m_ttbar, ticks_labels_E_ttbar, ticks_labels_dphi_ttbar, ticks_labels_Ht_ttbar, ticks_labels_yboost_ttbar, ticks_labels_ystar_ttbar, ticks_labels_chi_ttbar])



# ---------- FUNCTION DEFINITIONS ---------- #


# Imports data from 0th file and creates a data frame
# Input: reco_method (i.e.'KLFitter' or 'TRecNet' or 'TRecNet+ttbar') and file name
# Output: data frame
def Create_DF(reco_method,filename): 
	
    # Open first root file and its trees
    file0 = uproot.open('/data/jchishol/'+filename)
    tree_truth0 = file0['parton'].arrays()
    tree_reco0 = file0['reco'].arrays()

    # Create pandas dataframe
    if reco_method=='KLFitter':
        df = ak.to_pandas(tree_reco0['isMatched'])                              # Add reco isMatched to df
        df.rename(columns={'values':'reco_isMatched'},inplace=True)             # Rename first column appropriately
        df['truth_isDummy'] = ak.to_pandas(tree_truth0['isDummy'])              # Add truth isDummy to df 
        df['likelihood'] = ak.to_pandas(tree_reco0['klfitter_logLikelihood'])   # Add logLikelihood of klfitter
    
         # Add variables to be plotted
        for par in [top_had,top_lep,top_antitop]:         # For each particle
            for var in par.variables:                     # For each variable of that particle
                df['reco_'+par.particle+'_'+var] = ak.to_pandas(tree_reco0[level[par.particle][0]+var])      # Append the reco values
                df['truth_'+par.particle+'_'+var] = ak.to_pandas(tree_truth0[level[par.particle][1]+var])    # Append the true values

    else:
        df = ak.to_pandas(tree_reco0['ttbar_pt'])                # Create df
        df.rename(columns={'values':'reco_ttbar_pt'},inplace=True)             # Rename first column appropriately
        df['truth_ttbar_pt']=tree_truth0['ttbar_pt']

        # Add variables to be plotted
        for par in [top_had,top_lep,top_antitop]:                 # For each particle
            for var in par.variables:          # For each variable of that particle
                if par.particle=='ttbar' and var=='pt':   # We have alredy have ttbar_pt, so skip it
                    continue
                df['reco_'+par.particle+'_'+var] = ak.to_pandas(tree_reco0[par.particle+'_'+var])      # Append the reco values
                df['truth_'+par.particle+'_'+var] = ak.to_pandas(tree_truth0[par.particle+'_'+var])    # Append the true values

    # Close root file
    file0.close()

    print('Appended file '+filename)	

    # Return main data frame
    return df


# Appends data from the rest of the files to the existing data frame
# Note: currently used for KLFitter, not ML (only have one result file from the testing.ipynb)
# Input: data frame and name of file to be appended
# Output: data frame
def Append_Data(df,filename):

    # Open root file and its trees
    file_n = uproot.open('/data/jchishol/'+filename)
    tree_truth_n = file_n['parton'].arrays()
    tree_reco_n = file_n['reco'].arrays()

    # Create pandas data frame
    df_addon = ak.to_pandas(tree_reco_n['isMatched'])
    df_addon.rename(columns={'values':'reco_isMatched'},inplace=True)
    df_addon['truth_isDummy'] = ak.to_pandas(tree_truth_n['isDummy'])
    df_addon['likelihood'] = ak.to_pandas(tree_reco_n['klfitter_logLikelihood'])

    # Add variables to be plotted
    for par in [top_had,top_lep,top_antitop]:                     # For each particle
        for var in par.variables:          # For each variable of that particle
            df_addon['reco_'+par.particle+'_'+var] = ak.to_pandas(tree_reco_n[level[par.particle][0]+var])      # Append the reco values
            df_addon['truth_'+par.particle+'_'+var] = ak.to_pandas(tree_truth_n[level[par.particle][1]+var])    # Append the true values
        
    # Append data to the main data frame
    df = pd.concat([df,df_addon],ignore_index=True)

    # Close root file
    file_n.close()

    print('Appended file '+filename)

    return df


# Drops unmatched data, converts units, and calculates resolution
# Note: currently used for KLFitter, not ML
# Input: data frame
# Output: data frame
def Edit_Data(df):

    # Cut unwanted events
    indexNames = df[df['reco_isMatched'] == 0 ].index   # Want truth events that are detected in reco
    df.drop(indexNames,inplace=True)
    indexNames = df[df['truth_isDummy'] == 1 ].index    # Want reco events that match to an actual truth event
    df.drop(indexNames,inplace=True)  
    df.dropna(subset=['truth_th_y'],inplace=True)     # Don't want NaN events
    df.dropna(subset=['truth_tl_y'],inplace=True)     # Don't want NaN events

    for par in [top_had,top_lep,top_antitop]:           # For each particle
    	for var in par.variables:    # For each variable of that particle
        
            # Convert from MeV to GeV for some truth values
            if var in ['pt','m','E','pout','Ht']:             
                df['truth_'+par.particle+'_'+var] = df['truth_'+par.particle+'_'+var]/1000

    return df


# ------------- MAIN CODE ------------- #


for datatype in ['parton_ejets','parton_mjets']:
#for datatype in ['parton_ejets']:

    # Create the main data frame, append data from the rest of the files, fix it up, and then create a dataset object for KLFitter
    KLF_df = Create_DF('KLFitter','mc16e/mntuple_ttbar_0_'+datatype+'.root')
    for n in range(1,8):
        KLF_df = Append_Data(KLF_df,'mc16e/mntuple_ttbar_'+str(n)+'_'+datatype+'.root')
    KLF_df = Edit_Data(KLF_df)
    KLF_data = Plotter.Dataset(KLF_df,'KLFitter',datatype)

    # Create the data frame and dataset object for custom TRecNet model
    TRecNet_df = Create_DF('TRecNet','ML_Data/ML_Results_'+datatype+'_Model_Custom.root')
    TRecNet_data = Plotter.Dataset(TRecNet_df,'TRecNet',datatype)

    # Create the data frame and dataset object for custom+ttbar TRecNet model
    TRecNet_ttbar_df = Create_DF('TRecNet+ttbar','ML_Data/ML_Results_'+datatype+'_Model_Custom+ttbar.root')
    TRecNet_ttbar_data = Plotter.Dataset(TRecNet_ttbar_df,'TRecNet+ttbar',datatype)

    # Create plots
    for par in [top_had,top_lep,top_antitop]:
        for var in par.variables:
        
            # TruthReco Plots
            #Plotter.Plot.TruthReco_Hist(KLF_data,par,var,'./plots/'+datatype+'/'+par.particle+'/',30,LL_cut=-52)
            Plotter.Plot.TruthReco_Hist(TRecNet_data,par,var,'./plots/'+datatype+'/'+par.particle+'/',30)
            Plotter.Plot.TruthReco_Hist(TRecNet_ttbar_data,par,var,'./plots/'+datatype+'/'+par.particle+'/',30)
            
            # 2D Plots
            #Plotter.Plot.Normalized_2D(KLF_data,par,var,'./plots/'+datatype+'/'+par.particle+'/',color=plt.cm.Blues,LL_cut=-52)
            Plotter.Plot.Normalized_2D(TRecNet_data,par,var,'./plots/'+datatype+'/'+par.particle+'/',color=plt.cm.Purples)
            Plotter.Plot.Normalized_2D(TRecNet_ttbar_data,par,var,'./plots/'+datatype+'/'+par.particle+'/',color=colors.LinearSegmentedColormap.from_list('magenta_cmap', ['white', 'm']))
    
            # Res Plots
            res='Residuals' if var in ['eta','phi','y','pout','yboost','ystar','dphi'] else 'Resolution'
            Plotter.Plot.Res([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,var,'./plots/'+datatype+'/'+par.particle+'/',30,res,LL_cuts=[-52,-50,-48])

        # Res vs Var Plots
        #Plotter.Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'eta','eta',[-2.4,2.4,0.4],'./plots/'+datatype+'/'+par.particle+'/','Residuals',LL_cuts=[-52,-50,-48])
        #Plotter.Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'pt','pt',[90,510,30],'./plots/'+datatype+'/'+par.particle+'/','Resolution',LL_cuts=[-52,-50,-48])
        #Plotter.Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'pt','eta',[-2.4,2.4,0.4],'./plots/'+datatype+'/'+par.particle+'/','Resolution',LL_cuts=[-52,-50,-48])
        #Plotter.Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'eta','pt',[90,510,30],'./plots/'+datatype+'/'+par.particle+'/','Residuals',LL_cuts=[-52,-50,-48])

        #Plotter.Plot.Res_vs_Var([TRecNet_data,KLF_data],par,'phi','pt',[90,510,20],'./plots/'+datatype+'/'+par.particle+'/','Residuals',LL_cuts=[-52,-50,-48])
        #Plotter.Plot.Res_vs_Var([TRecNet_data,KLF_data],par,'pt','phi',[-3,3,0.5],'./plots/'+datatype+'/'+par.particle+'/','Resolution',LL_cuts=[-52,-50,-48])


print("done :)")
