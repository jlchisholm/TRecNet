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
from Plotter import *




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
                df['reco_'+par.name+'_'+var.name] = ak.to_pandas(tree_reco0[level[par.name][0]+var])      # Append the reco values
                df['truth_'+par.name+'_'+var.name] = ak.to_pandas(tree_truth0[level[par.name][1]+var])    # Append the true values

    else:
        df = ak.to_pandas(tree_reco0['ttbar_pt'])                # Create df
        df.rename(columns={'values':'reco_ttbar_pt'},inplace=True)             # Rename first column appropriately
        df['truth_ttbar_pt']=tree_truth0['ttbar_pt']

        # Add variables to be plotted
        for par in [top_had,top_lep,top_antitop]:                 # For each particle
            for var in par.variables:          # For each variable of that particle
                if par.name=='ttbar' and var.name=='pt':   # We have alredy have ttbar_pt, so skip it
                    continue
                df['reco_'+par.name+'_'+var.name] = ak.to_pandas(tree_reco0[par.name+'_'+var.name])      # Append the reco values
                df['truth_'+par.name+'_'+var.name] = ak.to_pandas(tree_truth0[par.name+'_'+var.name])    # Append the true values

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
            df_addon['reco_'+par.name+'_'+var.name] = ak.to_pandas(tree_reco_n[level[par.name][0]+var.name])      # Append the reco values
            df_addon['truth_'+par.name+'_'+var.name] = ak.to_pandas(tree_truth_n[level[par.name][1]+var.name])    # Append the true values
        
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
                df['truth_'+par.name+'_'+var.name] = df['truth_'+par.name+'_'+var.name]/1000

    return df


# ------------- MAIN CODE ------------- #



# List some useful strings, organized by particle
level = {'th' : ['klfitter_bestPerm_topHad_','MC_thad_afterFSR_'],
       'tl' : ['klfitter_bestPerm_topLep_','MC_tlep_afterFSR_'],
       'ttbar': ['klfitter_bestPerm_ttbar_','MC_ttbar_afterFSR_']}

# Define all the variables and their ranges we want
scale = '[GeV]'
pt = Variable('pt', 'p_T', np.arange(0,500,50), unit=scale)
eta = Variable('eta', '\eta', np.arange(-6,7.5,1.5))
y = Variable('y', 'y', np.arange(-2.5,3,0.5))
phi = Variable('phi', '\phi', np.arange(-3,3.75,0.75))
m_t = Variable('m','m', np.arange(100,260,20), unit=scale)
E_t = Variable('E','E', np.arange(100,1000,100), unit=scale)
pout_t = Variable('pout','p_{out}',range(-275,325,50), unit=scale)
m_tt = Variable('m','m', np.arange(200,1100,100), unit=scale)
E_tt = Variable('E','E', np.arange(200,2200,200), unit=scale)
dphi_tt = Variable('dphi','\Delta\phi', np.arange(0,4,0.5))
Ht_tt = Variable('Ht','H_T',np.arange(0,1100,100),unit=scale)
yboost_tt = Variable('yboost','y_{boost}',np.arange(-3,3.5,0.75))
ystar_tt = Variable('yboost','y_{star}',np.arange(-2.5,3,0.5))
chi_tt = Variable('chi','\chi',np.arange(0,25,2.5))

# Define the particles we want
top_had = Particle('th','t,had',[pt,eta,y,phi,m_t,E_t,pout_t])
top_lep = Particle('tl','t,lep',[pt,eta,y,phi,m_t,E_t,pout_t])
top_antitop = Particle('ttbar','t\overline{t}',[pt,eta,y,phi,m_tt,E_tt,dphi_tt,Ht_tt,yboost_tt,ystar_tt,chi_tt])






#for datatype in ['parton_ejets','parton_mjets']:
for datatype in ['parton_ejets']:

    # Create the main data frame, append data from the rest of the files, fix it up, and then create a dataset object for KLFitter
    #KLF_df = Create_DF('KLFitter','mc16e/mntuple_ttbar_0_'+datatype+'.root')
    #for n in range(1,8):
    #    KLF_df = Append_Data(KLF_df,'mc16e/mntuple_ttbar_'+str(n)+'_'+datatype+'.root')
    #KLF_df = Edit_Data(KLF_df)
    #KLF_data = Dataset(KLF_df,'KLFitter',datatype)

    # Create the data frame and dataset object for custom TRecNet model
    TRecNet_df = Create_DF('TRecNet','ML_Results/Model_Custom_Results_'+datatype+'.root')
    TRecNet_data = Dataset(TRecNet_df,'TRecNet',datatype)

    # Create the data frame and dataset object for custom+ttbar TRecNet model
    TRecNet_ttbar_df = Create_DF('TRecNet+ttbar','ML_Results/Model_Custom+ttbar_Results_'+datatype+'.root')
    TRecNet_ttbar_data = Dataset(TRecNet_ttbar_df,'TRecNet+ttbar',datatype)

    # Create the data frame and dataset object for jetPretrain+ttbar TRecNet model
    TRecNet_jetPretrain_df = Create_DF('TRecNet+ttbar','ML_Results/Model_JetPretrain+ttbar_Results_'+datatype+'.root')
    TRecNet_jetPretrain_data = Dataset(TRecNet_jetPretrain_df,'TRecNet_JetPretrain+ttbar',datatype)



    # # Create plots
    # for par in [top_had,top_lep,top_antitop]:
    #     for var in par.variables:
        
    #         # TruthReco Plots
    #         Plot.TruthReco_Hist(KLF_data,par,var,save_loc='./plots/'+datatype+'/'+par.particle+'/',nbins=30,LL_cut=-52)
    #         Plot.TruthReco_Hist(TRecNet_data,par,var,save_loc='./plots/'+datatype+'/'+par.particle+'/',nbins=30)
    #         Plot.TruthReco_Hist(TRecNet_ttbar_data,par,var,'./plots/'+datatype+'/'+par.particle+'/',n_bins=30)
            
    #         # 2D Plots
    #         Plot.Normalized_2D(KLF_data,par,var,save_loc='./plots/'+datatype+'/'+par.particle+'/',color=plt.cm.Blues,LL_cut=-52)
    #         Plot.Normalized_2D(TRecNet_data,par,var,save_loc='./plots/'+datatype+'/'+par.particle+'/',color=plt.cm.Purples)
    #         Plot.Normalized_2D(TRecNet_ttbar_data,par,var,save_loc='./plots/'+datatype+'/'+par.particle+'/',color=colors.LinearSegmentedColormap.from_list('magenta_cmap', ['white', 'm']))
    
    #         # Res Plots
    #         res='Residuals' if var in ['eta','phi','y','pout','yboost','ystar','dphi'] else 'Resolution'
    #         Plot.Res([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,var,save_loc='./plots/'+datatype+'/'+par.particle+'/',nbins=30,res=res,LL_cuts=[-52,-50,-48])

    #     # Res vs Var Plots
    #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'eta','eta',[-2.4,2.4,0.4],save_loc='./plots/'+datatype+'/'+par.particle+'/',y_res='Residuals',LL_cuts=[-52,-50,-48])
    #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'pt','pt',[90,510,30],save_loc='./plots/'+datatype+'/'+par.particle+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])
    #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'pt','eta',[-2.4,2.4,0.4],save_loc='./plots/'+datatype+'/'+par.particle+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])
    #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'eta','pt',[90,510,30],save_loc='./plots/'+datatype+'/'+par.particle+'/',y_res='Residuals',LL_cuts=[-52,-50,-48])
    #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'m','pt',[90,510,30],save_loc='./plots/'+datatype+'/'+par.particle+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])
    #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,KLF_data],par,'m','eta',[-2.4,2.4,0.4],save_loc='./plots/'+datatype+'/'+par.particle+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])

        #Plot.Res_vs_Var([TRecNet_data,KLF_data],par,'phi','pt',[90,510,20],save_loc='./plots/'+datatype+'/'+par.particle+'/',y_res='Residuals',LL_cuts=[-52,-50,-48])
        #Plot.Res_vs_Var([TRecNet_data,KLF_data],par,'pt','phi',[-3,3,0.5],save_loc='./plots/'+datatype+'/'+par.particle+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])


print("done :)")
