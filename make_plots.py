# Import useful packages
import uproot
import pandas as pd
import awkward as ak
import numpy as np
import vector
import itertools
import tk as tkinter
import matplotlib
matplotlib.use('Agg')  # need for not displaying plots when running batch jobs
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


    # For KLFitter or PseudoTop or Chi2:

    if reco_method=='KLFitter' or reco_method=='PseudoTop' or reco_method=='Chi2':

        tree = file0['nominal'].arrays()
        keys = file0['nominal'].keys()

        # Did an met_met cut for the ML data, so let's match it here too
        sel = tree['met_met']/1000 >= 20
        tree = tree[sel]

        # Initialize list of variable names
        names = []
        model_name = 'PseudoTop_Reco_' if reco_method=='PseudoTop' else 'TtresChi2_' if reco_method=='Chi2' else 'klfitter_bestPerm_'


        # Add reco variables to be plotted
        for par in [top_had,top_lep,top_antitop]:      # For each particle

            # Get all the possible names for this particle
            alt_par_names = list(par.alt_names)
            alt_par_names.append(par.name)

            for var in par.variables:                  # For each variable of that particle

                # Reset
                found_reco=False
                found_truth=False

                # Get all the possible names for this variable
                alt_var_names = list(var.alt_names)
                alt_var_names.append(var.name)

                for (par_name,var_name) in list(itertools.product(alt_par_names,alt_var_names)):
                    if model_name+par_name+'_'+var_name in keys:
                        names.append(['reco_'+par.name+'_'+var.name,model_name+par_name+'_'+var_name])
                        found_reco=True
                    if model_name+var_name in keys: # (particularly, chi_tt)
                        names.append(['reco_'+par.name+'_'+var.name,model_name+var_name])
                        found_reco=True
                    if 'MC_'+par_name+'_afterFSR_'+var_name in keys:
                        names.append(['truth_'+par.name+'_'+var.name,'MC_'+par_name+'_afterFSR_'+var_name])
                        found_truth=True
                    if 'MC_'+var_name in keys:  # (particularly, chi_tt)
                        names.append(['truth_'+par.name+'_'+var.name,'MC_'+var_name])
                        found_truth = True
                    # if 'MC_'+par_name+var_name in keys: # (in particular, for Pout)
                    #     df['truth_'+par.name+'_'+var.name] = ak.to_pandas(tree['MC_'+par_name+var_name])    # Append the true values
                    #     found_truth=True

                    # Stop looking for the correct names once we find them
                    if found_reco and found_truth:
                        break

        # Create dataframe!
        df = ak.to_pandas({new_name:tree[old_name] for (new_name,old_name) in names})

        # Bring in Chi2 for ... Chi2 lol
        if reco_method=='Chi2':
            df['chi2'] = tree['TtresChi2_Chi2']

        elif reco_method=='KLFitter':

            # Note: some events did not have a logLikelihood calculated -- we will need to pad these events to make sure nothing gets shifted weird, and then cut them
            df['logLikelihood'] = ak.flatten(ak.fill_none(ak.pad_none(tree['klfitter_logLikelihood'],1),np.nan))


        # Include number of jets, so we can look at how they might compare
        df['jet_n'] = tree['jet_n']

        # Drop bad klfitter events
        df.replace([np.inf,-np.inf],np.nan,inplace=True)
        df.dropna(inplace=True)



    # For TRecNet models:

    else:
        tree_truth0 = file0['parton'].arrays()
        tree_reco0 = file0['reco'].arrays()
        keys = file0['parton'].keys()   # Should be same for reco

        truth_df = ak.to_pandas({'truth_'+key:tree_truth0[key] for key in keys})
        reco_df = ak.to_pandas({'reco_'+key:tree_reco0[key] for key in keys})
        df = pd.concat([truth_df,reco_df], axis=1)

    # Close root file
    file0.close()

    print('Appended file '+filename)	

    # Return main data frame
    return df


# Appends data from the rest of the files to the existing data frame
# Note: currently used for KLFitter, not ML (only have one result file from the testing.ipynb)
# Input: data frame and name of file to be appended
# Output: data frame
def Append_Data(df,reco_method,filename):

    # Get the data from the new file
    df_addon = Create_DF(reco_method,filename)
        
    # Append data to the main data frame
    df = pd.concat([df,df_addon],axis=0,ignore_index=True)

    #print('Appended file '+filename)

    return df


# KLFitter is missing some variables I'd like to compare

def Append_Calculations(df,reco):

    for level in ['truth','reco']:

        # ttbar deta
        th_vec = vector.arr({"pt": df[level+'_th_pt'], "phi": df[level+'_th_phi'], "eta": df[level+'_th_eta'],"mass": df[level+'_th_m']})
        tl_vec = vector.arr({"pt": df[level+'_tl_pt'], "phi": df[level+'_tl_phi'], "eta": df[level+'_tl_eta'],"mass": df[level+'_tl_m']}) 
        df[level+'_ttbar_deta'] = abs(th_vec.deltaeta(tl_vec))

        if reco=='KLFitter' or reco=='PseudoTop':

            # dphi
            df[level+'_ttbar_dphi'] = abs(th_vec.deltaphi(tl_vec))

            # ttbar Ht
            df[level+'_ttbar_Ht'] = df[level+'_th_pt']+df[level+'_tl_pt']

            # ttbar yboost and ystar
            df[level+'_ttbar_yboost'] = 0.5*(df[level+'_th_y']+df[level+'_tl_y'])
            #df[level+'_ttbar_ystar'] = 0.5*(df[level+'_th_y']-df[level+'_tl_y'])


    return df






# I THINK THIS IS NOW ALL DONE IN THE PREP STAGE
# CONFUSED BECUZ PRETTY SURE I DID BUT IT'S NOT REALLY THERE
# NVM CUTS ARE DONE BUT NOT SCALING NEED TO DO THAT NOW

# Drops unmatched data, converts units, and calculates resolution
# Note: currently used for KLFitter, not ML
# Input: data frame
# Output: data frame
def Edit_Data(df):

    # Cut unwanted events
    #indexNames = df[df['reco_isMatched'] == 0 ].index   # Want truth events that are detected in reco
    #df.drop(indexNames,inplace=True)
    #indexNames = df[df['truth_isDummy'] == 1 ].index    # Want reco events that match to an actual truth event
    #df.drop(indexNames,inplace=True)  
    #df.dropna(subset=['truth_th_y'],inplace=True)     # Don't want NaN events
    #df.dropna(subset=['truth_tl_y'],inplace=True)     # Don't want NaN events

    for par in [top_had,top_lep,top_antitop]:           # For each particle
    	for var in par.variables:    # For each variable of that particle
        
            # Convert from MeV to GeV for some truth values
            if var.name in ['pt','m','E','pout','Ht'] and 'truth_'+par.name+'_'+var.name in df.keys():  
                df['truth_'+par.name+'_'+var.name] = df['truth_'+par.name+'_'+var.name]/1000

    return df



def Check(df,reco):

    print('Checking ...',reco)

    for level in ['truth','reco']:
    
        # Check pt, m, and E are not negative, and phi is within (-pi, pi)
        for par in ['th', 'tl', 'ttbar']:
            for var in ['pt','m','E']:
                if min(df[level+'_'+par+'_'+var]) < 0:
                    print('WARNING: '+reco+' has '+par+'_'+var+' < 0 at the '+level+' level!')
                    print(df[df[level+'_'+par+'_'+var]<0])
            if max(df[level+'_'+par+'_phi']) > np.pi:
                print('WARNING: '+reco+' has '+par+'_phi'+' > pi at the '+level+' level!')  
                print(df[df[level+'_'+par+'_phi'] > np.pi])
            if min(df[level+'_'+par+'_phi']) < -np.pi:
                print('WARNING: '+reco+' has '+par+'_phi'+' < -pi at the '+level+' level!')    
                print(df[df[level+'_'+par+'_phi'] < -np.pi])    

        # Check Ht, chi, dphi, deta are all above zero
        for var in ['Ht','chi','dphi','deta']:
            if min(df[level+'_ttbar_'+var]) < 0:
                print('WARNING: '+reco+' has ttbar_'+var+' < 0 at the '+level+' level!')  
                print(df[df[level+'_ttbar_'+var] < 0])       

        # Check ttbar_dphi < pi
        if max(df[level+'_ttbar_dphi']) > np.pi:
            print('WARNING: '+reco+' has ttbar_dphi > pi at the '+level+' level!')  
            print(df[df[level+'_ttbar_dphi'] > np.pi])

        

            
    


# ------------- MAIN CODE ------------- #




# Define all the variables and their ranges we want
scale = '[GeV]'
pt = Variable('pt', 'p_T', np.arange(0,550,50), unit=scale)
eta = Variable('eta', '\eta', np.arange(-6,7.5,1.5))
y = Variable('y', 'y', np.arange(-2.5,3,0.5))
phi = Variable('phi', '\phi', np.arange(-3,3.75,0.75))
m_t = Variable('m','m', np.arange(100,260,20),unit=scale)
E_t = Variable('E','E', np.arange(100,1600,150),unit=scale)
#pout_t = Variable('pout','p_{out}',range(-275,325,50), unit=scale,alt_names=['Pout'])
m_tt = Variable('m','m', np.arange(200,1700,150),unit=scale)
E_tt = Variable('E','E', np.arange(200,2700,250),unit=scale)
dphi_tt = Variable('dphi','|\Delta\phi|', np.arange(0,3.6,0.45),alt_names=['deltaPhi'])
deta_tt = Variable('deta','|\Delta\eta|', np.arange(0,10,1),alt_names=['deltaEta'])
Ht_tt = Variable('Ht','H_T',np.arange(0,1200,120),unit=scale,alt_names=['HT'])
yboost_tt = Variable('yboost','y_{boost}',np.arange(-3,3.75,0.75),alt_names=['y_boost'])
#ystar_tt = Variable('ystar','y_{star}',np.arange(-2.5,3,0.5),alt_names=['y_star'])
chi_tt = Variable('chi','\chi',np.arange(0,25,2.5),alt_names=['chi_tt'])

# Define the particles we want
top_had = Particle('th','t,had',[pt,eta,y,phi,m_t,E_t],alt_names=['thad','topHad','top_had'])
top_lep = Particle('tl','t,lep',[pt,eta,y,phi,m_t,E_t],alt_names=['tlep','topLep','top_lep'])
top_antitop = Particle('ttbar','t\overline{t}',[pt,eta,y,phi,m_tt,E_tt,dphi_tt,deta_tt,Ht_tt,yboost_tt,chi_tt])






#for datatype in ['parton_ejets','parton_mjets']:
for datatype in ['parton_e+mjets']:   # Should be able to remove this now

    # Create the data frames and dataset objects for KLFitter, PseudoTop
    # KLF_df = Create_DF('KLFitter','mc1516_new/mntuple_ljets_01_jetMatch04.root')
    # PT_df = Create_DF('PseudoTop','mc1516_new/mntuple_ljets_01_jetMatch04.root')
    # Chi2_df = Create_DF('Chi2','mc1516_new/mntuple_ljets_01_jetMatch04.root')


    # for n in ['02','03','04']:  # Use the same files as were used for testing!
    #     KLF_df = Append_Data(KLF_df,'KLFitter','mc1516_new/mntuple_ljets_'+n+'_jetMatch04.root')
    #     PT_df = Append_Data(PT_df,'PseudoTop','mc1516_new/mntuple_ljets_'+n+'_jetMatch04.root')
    #     Chi2_df = Append_Data(Chi2_df,'Chi2','mc1516_new/mntuple_ljets_'+n+'_jetMatch04.root')

    # KLF_df = Append_Calculations(KLF_df,'KLFitter')
    # KLF_df = Edit_Data(KLF_df)
    # Check(KLF_df,'KLFitter')
    # KLF_data = Dataset(KLF_df,'KLFitter',datatype,'tab:blue')
    # Cut_KLF_df = KLF_df[KLF_df['logLikelihood']>-52]
    # Cut_KLF_data = Dataset(Cut_KLF_df,'KLFitter',datatype,'tab:cyan',cuts='LL>-52')


    # PT_df = Append_Calculations(PT_df,'PseudoTop')
    # PT_df = Edit_Data(PT_df)
    # Check(PT_df,'PseudoTop')
    # PT_data = Dataset(PT_df,'PseudoTop',datatype,'tab:red')

    # #Chi2_df = Append_Calculations(Chi2_df,'Chi2')
    # Chi2_df = Edit_Data(Chi2_df)
    # Check(Chi2_df,'Chi2')
    # Chi2_data = Dataset(Chi2_df,'Chi2',datatype,'tab:green')
    # Cut_Chi2_df = Chi2_df[Chi2_df['chi2']<50]
    # Cut_Chi2_data = Dataset(Cut_Chi2_df,'Chi2',datatype,'tab:olive','chi2<50')





    # Create the data frame and dataset object for custom TRecNet model
    TRecNet_df = Create_DF('TRecNet','ML_Results/Model_Custom_new_Results.root')
    TRecNet_df = Append_Calculations(TRecNet_df,'TRecNet')
    Check(TRecNet_df,'TRecNet')
    TRecNet_data = Dataset(TRecNet_df,'TRecNet',datatype,'tab:orange')

    # Create the data frame and dataset object for custom+ttbar TRecNet model
    TRecNet_ttbar_df = Create_DF('TRecNet+ttbar','ML_Results/Model_Custom+ttbar_new_Results.root')
    TRecNet_ttbar_df = Append_Calculations(TRecNet_ttbar_df,'TRecNet+ttbar')
    Check(TRecNet_ttbar_df,'TRecNet+ttbar')
    TRecNet_ttbar_data = Dataset(TRecNet_ttbar_df,'TRecNet+ttbar',datatype,'tab:pink')

    # Create the data frame and dataset object for jetPretrain+ttbar TRecNet model
    TRecNet_jetPretrain_df = Create_DF('TRecNet+ttbar','ML_Results/Model_JetPretrain+ttbar_Results.root')
    TRecNet_jetPretrain_df = Append_Calculations(TRecNet_jetPretrain_df,'TRecNet_JetPretrain+ttbar')
    Check(TRecNet_jetPretrain_df,'TRecNet_jetPretrain+ttbar')
    TRecNet_jetPretrain_data = Dataset(TRecNet_jetPretrain_df,'TRecNet_JetPretrain+ttbar',datatype,'tab:purple')


    # Create plots
    # for par in [top_had,top_lep,top_antitop]:
    #     for var in par.variables:
        
    #         # TruthReco Plots
    #         Plot.TruthReco_Hist(KLF_data,par,var,save_loc='new_plots/'+par.name+'/',nbins=30)
    #         Plot.TruthReco_Hist(Cut_KLF_data,par,var,save_loc='new_plots/'+par.name+'/',nbins=30)
    #         Plot.TruthReco_Hist(PT_data,par,var,save_loc='new_plots/'+par.name+'/',nbins=30)
    #         Plot.TruthReco_Hist(TRecNet_data,par,var,save_loc='new_plots/'+par.name+'/',nbins=30)
    #         Plot.TruthReco_Hist(TRecNet_ttbar_data,par,var,'new_plots/'+par.name+'/',nbins=30)
    #         Plot.TruthReco_Hist(TRecNet_jetPretrain_data,par,var,'new_plots/'+par.name+'/',nbins=30)
            
    #         # 2D Plots
    #         Plot.Confusion_Matrix(KLF_data,par,var,save_loc='new_plots/'+par.name+'/')
    #         Plot.Confusion_Matrix(Cut_KLF_data,par,var,save_loc='new_plots/'+par.name+'/')
    #         Plot.Confusion_Matrix(PT_data,par,var,save_loc='new_plots/'+par.name+'/')
    #         Plot.Confusion_Matrix(TRecNet_data,par,var,save_loc='new_plots/'+par.name+'/')
    #         Plot.Confusion_Matrix(TRecNet_ttbar_data,par,var,save_loc='new_plots/'+par.name+'/')
    #         Plot.Confusion_Matrix(TRecNet_jetPretrain_data,par,var,save_loc='new_plots/'+par.name+'/')

    #         # Chi2 doesn't have all the same variables
    #         if 'reco_'+par.name+'_'+var.name in Chi2_data.df.keys():
    #             Plot.TruthReco_Hist(Chi2_data,par,var,save_loc='new_plots/'+par.name+'/',nbins=30)
    #             Plot.TruthReco_Hist(Cut_Chi2_data,par,var,save_loc='new_plots/'+par.name+'/',nbins=30)
    #             Plot.Confusion_Matrix(Chi2_data,par,var,save_loc='new_plots/'+par.name+'/')
    #             Plot.Confusion_Matrix(Cut_Chi2_data,par,var,save_loc='new_plots/'+par.name+'/')

    #             # Res Plots (Note: ml2 seems to crash when we use cauchy as the core fit option now :/)
    #             res_type='Residuals' if var.name in ['eta','phi','y','pout','yboost','ystar','dphi','deta'] else 'Resolution'
    #             Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF_data,Cut_KLF_data],par,var,save_loc='new_plots/'+par.name+'/',nbins=30,res=res_type,core_fit='nofit')

    #         else:
    #             # Res Plots
    #             res_type='Residuals' if var.name in ['eta','phi','y','pout','yboost','ystar','dphi','deta'] else 'Resolution'
    #             Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF_data,Cut_KLF_data],par,var,save_loc='new_plots/'+par.name+'/',nbins=30,res=res_type,core_fit='nofit')

    #     # Res vs Var Plots (again chi2 doesn't have as much)
    #     if par.name=='ttbar':
    #         Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF_data,Cut_KLF_data],par,eta,eta,[-2.4,2.4,0.4],save_loc='new_plots/'+par.name+'/',y_res='Residuals',core_fit='nofit')
    #         Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF_data,Cut_KLF_data],par,pt,pt,[90,510,30],save_loc='new_plots/'+par.name+'/',y_res='Resolution',core_fit='nofit')
    #         Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF_data,Cut_KLF_data],par,pt,eta,[-2.4,2.4,0.4],save_loc='new_plots/'+par.name+'/',y_res='Resolution',core_fit='nofit')
    #         Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF_data,Cut_KLF_data],par,eta,pt,[90,510,30],save_loc='new_plots/'+par.name+'/',y_res='Residuals',core_fit='nofit')
    #     else:
    #         Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF_data,Cut_KLF_data],par,eta,eta,[-2.4,2.4,0.4],save_loc='new_plots/'+par.name+'/',y_res='Residuals',core_fit='nofit')
    #         Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF_data,Cut_KLF_data],par,pt,pt,[90,510,30],save_loc='new_plots/'+par.name+'/',y_res='Resolution',core_fit='nofit')
    #         Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF_data,Cut_KLF_data],par,pt,eta,[-2.4,2.4,0.4],save_loc='new_plots/'+par.name+'/',y_res='Resolution',core_fit='nofit')
    #         Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF_data,Cut_KLF_data],par,eta,pt,[90,510,30],save_loc='new_plots/'+par.name+'/',y_res='Residuals',core_fit='nofit')


        #Plot.Res_vs_Var([KLF_data, TRecNet_ttbar_data,TRecNet_jetPretrain_data],par,'m','pt',[90,510,30],save_loc='new_plots/'+par.name+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])
        #Plot.Res_vs_Var([KLF_data, TRecNet_ttbar_data,TRecNet_jetPretrain_data],par,'m','eta',[-2.4,2.4,0.4],save_loc='new_plots/'+par.name+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])

        # Plot.Res_vs_Var([TRecNet_jetPretrain_data],par,'phi','pt',[90,510,20],save_loc='.',y_res='Residuals',LL_cuts=[-52,-50,-48])
        # Plot.Res_vs_Var([TRecNet_jetPretrain_data],par,'pt','phi',[-3,3,0.5],save_loc='.',y_res='Resolution',LL_cuts=[-52,-50,-48])


print("done :)")
