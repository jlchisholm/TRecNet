# Import useful packages
import uproot
import h5py
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


# Reads the data into a dataframe, and removes events that are not part of the test dataset
# Input: reco_method (i.e.'KLFitter', 'TRecNet', etc.), file name, and list of the event numbers for the test datset
# Output: data frame
def readInData(reco_method,filename,eventnumbers=[]):

    print('Reading in data for',reco_method)

    # Open first root file and its trees
    file0 = uproot.open('/mnt/xrootdg/jchishol/mntuples_08_01_22/'+filename)

    if 'TRecNet' in reco_method:
        tree_truth0 = file0['parton'].arrays()
        tree_reco0 = file0['reco'].arrays()
        keys = file0['parton'].keys()   # Should be same for reco

        truth_df = ak.to_pandas({'truth_'+key:tree_truth0[key] for key in keys})
        reco_df = ak.to_pandas({'reco_'+key:tree_reco0[key] for key in keys})
        df = pd.concat([truth_df,reco_df], axis=1)

        df['eventNumber'] = tree_truth0['eventNumber']

    
    else:

        # Reco is a misnomer (for KLF, etc, but not systematics), since truth observables are in there too -- should change this later maybe
        tree = file0['reco'].arrays()
        keys = file0['reco'].keys()

        # Only take events with the same event numbers as the test data
        sel = np.isin(tree['eventNumber'],eventnumbers)
        tree = tree[sel]

        # Create dataframe!
        df = ak.to_pandas({key:tree[key] for key in keys})

    # Bit of a hack for now:
    if reco_method=='sys':
        df = df.add_prefix('reco_')
        df = df.rename(columns={'reco_eventNumber':'eventNumber'})

    return df



# KLFitter is missing some observables I'd like to compare

def Append_Calculations(df,reco):

    for level in ['truth','reco']:

        if level=='truth' and ('sys' in reco):
            continue

        df[level+'_ttbar_px'] = df[level+'_ttbar_pt']*np.cos(df[level+'_ttbar_phi'])
        df[level+'_ttbar_py'] = df[level+'_ttbar_pt']*np.sin(df[level+'_ttbar_phi'])

        

        if reco!='Chi2':

            df[level+'_th_px'] = df[level+'_th_pt']*np.cos(df[level+'_th_phi'])
            df[level+'_th_py'] = df[level+'_th_pt']*np.sin(df[level+'_th_phi'])
            df[level+'_tl_px'] = df[level+'_tl_pt']*np.cos(df[level+'_tl_phi'])
            df[level+'_tl_py'] = df[level+'_tl_pt']*np.sin(df[level+'_tl_phi'])

            # ttbar deta
            th_vec = vector.arr({"pt": df[level+'_th_pt'], "phi": df[level+'_th_phi'], "eta": df[level+'_th_eta'],"mass": df[level+'_th_m']})
            tl_vec = vector.arr({"pt": df[level+'_tl_pt'], "phi": df[level+'_tl_phi'], "eta": df[level+'_tl_eta'],"mass": df[level+'_tl_m']}) 
            df[level+'_ttbar_deta'] = abs(th_vec.deltaeta(tl_vec))

            if reco=='KLFitter4' or reco=='KLFitter6' or reco=='PseudoTop':

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
    	for var in par.observables:    # For each observable of that particle
        
            # Convert from MeV to GeV for some truth values
            if var.name in ['pt','px','py','m','E','pout','Ht'] and 'truth_'+par.name+'_'+var.name in df.keys():  
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
                print(df[df[level+'_'+par+'_phi'] > np.pi][level+'_'+par+'_phi'])
                print(df[df[level+'_'+par+'_phi'] > np.pi]['eventNumber'])
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




# Define all the observables and their ranges we want
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






#for datatype in ['parton_ejets','parton_mjets']:
for datatype in ['parton_e+mjets']:   # Should be able to remove this now


    # Get the total number of test events and the list of test events
    f_test = h5py.File('/mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_jetMatch04_6jets_test.h5','r')
    eventnumbers = np.array(f_test.get('eventNumber'))
    total_test_events = len(np.array(f_test.get('eventNumber')))
    f_test.close()

    # Read in the TRecNet dataframes
    TRecNet_df = readInData('TRecNet','Model_Custom_full_Results.root')
    TRecNet_ttbar_df = readInData('TRecNet+ttbar','Model_Custom+ttbar_full_Results.root')
    TRecNet_jetPretrain_df = readInData('TRecNet+ttbar+jetPretrain','Model_JetPretrain+ttbar_full_Results.root')
    
    # Read in systematics for TRecNet
    # TRecNet_sysUP_df = readInData('sys','Model_Custom_sysUP_Results.root',eventnumbers)
    # TRecNet_sysDOWN_df = readInData('sys','Model_Custom_sysDOWN_Results.root',eventnumbers)
    # TRecNet_ttbar_sysUP_df = readInData('sys','Model_Custom+ttbar_sysUP_Results.root',eventnumbers)
    # TRecNet_ttbar_sysDOWN_df = readInData('sys','Model_Custom+ttbar_sysDOWN_Results.root',eventnumbers)
    # TRecNet_jetPretrain_sysUP_df = readInData('sys','Model_JetPretrain+ttbar_sysUP_Results.root',eventnumbers)
    # TRecNet_jetPretrain_sysDOWN_df = readInData('sys','Model_JetPretrain+ttbar_sysDOWN_Results.root',eventnumbers)

    # Read in the data frames for KLFitter, PseudoTop, and Chi2
    #KLF4_df = readInData('KLFitter4','KLF4_Results.root',eventnumbers)
    KLF6_df = readInData('KLFitter6','KLF6_Results.root',eventnumbers)
    PT_df = readInData('PseudoTop','PT_Results.root',eventnumbers)
    Chi2_df = readInData('Chi2','Chi2_Results.root',eventnumbers)

    # Append calculations of some observables
    TRecNet_df = Append_Calculations(TRecNet_df,'TRecNet')
    TRecNet_ttbar_df = Append_Calculations(TRecNet_ttbar_df,'TRecNet+ttbar')
    TRecNet_jetPretrain_df = Append_Calculations(TRecNet_jetPretrain_df,'TRecNet_JetPretrain+ttbar')

    # TRecNet_sysUP_df = Append_Calculations(TRecNet_sysUP_df,'TRecNet_sysUP')
    # TRecNet_ttbar_sysUP_df = Append_Calculations(TRecNet_ttbar_sysUP_df,'TRecNet+ttbar_sysUP')
    # TRecNet_jetPretrain_sysUP_df = Append_Calculations(TRecNet_jetPretrain_sysUP_df,'TRecNet_JetPretrain+ttbar_sysUP')
    # TRecNet_sysDOWN_df = Append_Calculations(TRecNet_sysDOWN_df,'TRecNet_sysDOWN')
    # TRecNet_ttbar_sysDOWN_df = Append_Calculations(TRecNet_ttbar_sysDOWN_df,'TRecNet+ttbar_sysDOWN')
    # TRecNet_jetPretrain_sysDOWN_df = Append_Calculations(TRecNet_jetPretrain_sysDOWN_df,'TRecNet_JetPretrain+ttbar_sysDOWN')

    # KLF4_df = Append_Calculations(KLF4_df,'KLFitter4')
    KLF6_df = Append_Calculations(KLF6_df,'KLFitter6')
    PT_df = Append_Calculations(PT_df,'PseudoTop')
    Chi2_df = Append_Calculations(Chi2_df,'Chi2')

    # Edit data as necessary
    # KLF4_df = Edit_Data(KLF4_df)
    KLF6_df = Edit_Data(KLF6_df)
    PT_df = Edit_Data(PT_df)
    Chi2_df = Edit_Data(Chi2_df)

    # Do some checks
    #Check(TRecNet_df,'TRecNet')
    #Check(TRecNet_ttbar_df,'TRecNet+ttbar')
    #Check(TRecNet_jetPretrain_df,'TRecNet_jetPretrain+ttbar')
    #Check(KLF4_df,'KLFitter')
    #Check(PT_df,'PseudoTop')
    #Check(Chi2_df,'Chi2')


    # TRecNet_sysUP_data = Dataset(TRecNet_sysUP_df,'TRecNet',datatype,'tab:gray',sys_type='UP')
    # TRecNet_ttbar_sysUP_data = Dataset(TRecNet_ttbar_sysUP_df,'TRecNet+ttbar',datatype,'tab:pink',sys='UP')
    # TRecNet_jetPretrain_sysUP_data = Dataset(TRecNet_jetPretrain_sysUP_df,'TRecNet_JetPretrain+ttbar',datatype,'tab:purple',sys='UP')
    # TRecNet_sysDOWN_data = Dataset(TRecNet_sysDOWN_df,'TRecNet',datatype,'tab:brown',sys_type='DOWN')
    # TRecNet_ttbar_sysDOWN_data = Dataset(TRecNet_ttbar_sysDOWN_df,'TRecNet+ttbar',datatype,'tab:pink',sys='DOWN')
    # TRecNet_jetPretrain_sysDOWN_data = Dataset(TRecNet_jetPretrain_sysDOWN_df,'TRecNet_JetPretrain+ttbar',datatype,'tab:purple',sys='DOWN')

    # Create the datasets without systematics
    TRecNet_data = Dataset(TRecNet_df,'TRecNet',datatype,'tab:orange')
    TRecNet_ttbar_data = Dataset(TRecNet_ttbar_df,'TRecNet+ttbar',datatype,'tab:pink')
    TRecNet_jetPretrain_data = Dataset(TRecNet_jetPretrain_df,'TRecNet_JetPretrain+ttbar',datatype,'tab:purple')

    # OR create the datasets WITH systematics
    # TRecNet_data = Dataset(TRecNet_df,'TRecNet',datatype,'tab:orange',systematics=[TRecNet_sysDOWN_data,TRecNet_sysUP_data])
    # TRecNet_ttbar_data = Dataset(TRecNet_ttbar_df,'TRecNet+ttbar',datatype,'tab:pink',systematics=[TRecNet_ttbar_sysDOWN_data,TRecNet_ttbar_sysUP_data])
    # TRecNet_jetPretrain_data = Dataset(TRecNet_jetPretrain_df,'TRecNet_JetPretrain+ttbar',datatype,'tab:purple',systematics=[TRecNet_jetPretrain_sysDOWN_data,TRecNet_jetPretrain_sysUP_data])



    # KLF4_data = Dataset(KLF4_df,'KLFitter4',datatype,'tab:blue',perc_events=int(100*len(KLF4_df)/total_test_events))
    # Cut_KLF4_df = KLF4_df[KLF4_df['logLikelihood']>-52]
    # Cut_KLF4_data = Dataset(Cut_KLF4_df,'KLFitter4',datatype,'tab:cyan',cuts='LL>-52',perc_events=int(100*len(Cut_KLF4_df)/total_test_events))

    KLF6_data = Dataset(KLF6_df,'KLFitter6',datatype,'tab:blue',perc_events=int(100*len(KLF6_df)/total_test_events))
    Cut_KLF6_df = KLF6_df[KLF6_df['logLikelihood']>-52]
    Cut_KLF6_data = Dataset(Cut_KLF6_df,'KLFitter6',datatype,'tab:cyan',cuts='LL>-52',perc_events=int(100*len(Cut_KLF6_df)/total_test_events))

    PT_data = Dataset(PT_df,'PseudoTop',datatype,'tab:red',perc_events=int(100*len(PT_df)/total_test_events))

    Chi2_data = Dataset(Chi2_df,'Chi2',datatype,'tab:green',perc_events=int(100*len(Chi2_df)/total_test_events))
    Cut_Chi2_df = Chi2_df[Chi2_df['chi2']<50]
    Cut_Chi2_data = Dataset(Cut_Chi2_df,'Chi2',datatype,'tab:olive',cuts='chi2<50',perc_events=int(100*len(Cut_Chi2_df)/total_test_events))

    
    

    # Easier to set directories here
    dir = 'full_plots/'

    # Create plots
    for par in [top_had,top_lep,top_antitop]:

        for var in par.observables:
             
            for ds in [KLF6_data,Cut_KLF6_data,PT_data,TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data]:

                Plot.TruthReco_Hist(ds,par,var,save_loc=dir+par.name+'/TruthReco/',nbins=30)
                # Plot.Confusion_Matrix(ds,par,var,save_loc=dir+par.name+'/CM/')
                # Plot.Confusion_Matrix(ds,par,var,save_loc=dir+par.name+'/CM/',pt_low=0,pt_high=100)
                # Plot.Confusion_Matrix(ds,par,var,save_loc=dir+par.name+'/CM/',pt_low=100,pt_high=250)
                # Plot.Confusion_Matrix(ds,par,var,save_loc=dir+par.name+'/CM/',pt_low=250)


            # # Chi2 doesn't have all the same observables
            # if 'reco_'+par.name+'_'+var.name in Chi2_data.df.keys():
            #     #Plot.TruthReco_Hist(Chi2_data,par,var,save_loc=dir+par.name+'/TruthReco/',nbins=30)
            #     #Plot.TruthReco_Hist(Cut_Chi2_data,par,var,save_loc=dir+par.name+'/TruthReco/',nbins=30)

            #     Plot.Confusion_Matrix(Chi2_data,par,var,save_loc=dir+par.name+'/CM/')
            #     Plot.Confusion_Matrix(Cut_Chi2_data,par,var,save_loc=dir+par.name+'/CM/')

            #     Plot.Confusion_Matrix(Chi2_data,par,var,save_loc=dir+par.name+'/CM/',pt_low=0,pt_high=100)
            #     Plot.Confusion_Matrix(Cut_Chi2_data,par,var,save_loc=dir+par.name+'/CM/',pt_low=0,pt_high=100)
            #     Plot.Confusion_Matrix(Chi2_data,par,var,save_loc=dir+par.name+'/CM/',pt_low=100,pt_high=250)
            #     Plot.Confusion_Matrix(Cut_Chi2_data,par,var,save_loc=dir+par.name+'/CM/',pt_low=100,pt_high=250)
            #     Plot.Confusion_Matrix(Chi2_data,par,var,save_loc=dir+par.name+'/CM/',pt_low=250)
            #     Plot.Confusion_Matrix(Cut_Chi2_data,par,var,save_loc=dir+par.name+'/CM/',pt_low=250)

                # # Res Plots (Note: ml2 seems to crash when we use cauchy as the core fit option now :/)
                # res_type='Residuals' if var.name in ['eta','phi','y','pout','yboost','ystar','dphi','deta'] else 'Resolution'
                # Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF6_data,Cut_KLF6_data],par,var,save_loc=dir+par.name+'/Res/',nbins=30,res=res_type,core_fit='nofit')
                # Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF6_data,Cut_KLF6_data],par,var,save_loc=dir+par.name+'/Res/',nbins=30,res=res_type,core_fit='nofit',pt_low=0,pt_high=100)
                # Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF6_data,Cut_KLF6_data],par,var,save_loc=dir+par.name+'/Res/',nbins=30,res=res_type,core_fit='nofit',pt_low=100,pt_high=250)
                # Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF6_data,Cut_KLF6_data],par,var,save_loc=dir+par.name+'/Res/',nbins=30,res=res_type,core_fit='nofit',pt_low=250)

        #     else:
        #         # Res Plots
        #         res_type='Residuals' if var.name in ['eta','phi','y','pout','yboost','ystar','dphi','deta'] else 'Resolution'
        #         Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF6_data,Cut_KLF6_data],par,var,save_loc=dir+par.name+'/Res/',nbins=30,res=res_type,core_fit='nofit')
        #         Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF6_data,Cut_KLF6_data],par,var,save_loc=dir+par.name+'/Res/',nbins=30,res=res_type,core_fit='nofit',pt_low=0,pt_high=100)
        #         Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF6_data,Cut_KLF6_data],par,var,save_loc=dir+par.name+'/Res/',nbins=30,res=res_type,core_fit='nofit',pt_low=100,pt_high=250)
        #         Plot.Res([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF6_data,Cut_KLF6_data],par,var,save_loc=dir+par.name+'/Res/',nbins=30,res=res_type,core_fit='nofit',pt_low=250)

        # # Res vs Var Plots (again chi2 doesn't have as much)
        # if par.name=='ttbar':
        #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF6_data,Cut_KLF6_data],par,eta,eta,[-2.4,2.4,0.4],save_loc=dir+par.name+'/Res/',y_res='Residuals',core_fit='nofit')
        #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF6_data,Cut_KLF6_data],par,pt,pt,[0,600,30],save_loc=dir+par.name+'/Res/',y_res='Resolution',core_fit='nofit')
        #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF6_data,Cut_KLF6_data],par,pt,eta,[-2.4,2.4,0.4],save_loc=dir+par.name+'/Res/',y_res='Resolution',core_fit='nofit')
        #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,Chi2_data,Cut_Chi2_data,KLF6_data,Cut_KLF6_data],par,eta,pt,[0,600,30],save_loc=dir+par.name+'/Res/',y_res='Residuals',core_fit='nofit')
        # else:
        #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF6_data,Cut_KLF6_data],par,eta,eta,[-2.4,2.4,0.4],save_loc=dir+par.name+'/Res/',y_res='Residuals',core_fit='nofit')
        #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF6_data,Cut_KLF6_data],par,pt,pt,[0,600,30],save_loc=dir+par.name+'/Res/',y_res='Resolution',core_fit='nofit')
        #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF6_data,Cut_KLF6_data],par,pt,eta,[-2.4,2.4,0.4],save_loc=dir+par.name+'/Res/',y_res='Resolution',core_fit='nofit')
        #     Plot.Res_vs_Var([TRecNet_data,TRecNet_ttbar_data,TRecNet_jetPretrain_data,PT_data,KLF6_data,Cut_KLF6_data],par,eta,pt,[0,600,30],save_loc=dir+par.name+'/Res/',y_res='Residuals',core_fit='nofit')


        #Plot.Res_vs_Var([KLF_data, TRecNet_ttbar_data,TRecNet_jetPretrain_data],par,'m','pt',[90,510,30],save_loc=dir+par.name+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])
        #Plot.Res_vs_Var([KLF_data, TRecNet_ttbar_data,TRecNet_jetPretrain_data],par,'m','eta',[-2.4,2.4,0.4],save_loc=dir+par.name+'/',y_res='Resolution',LL_cuts=[-52,-50,-48])

        # Plot.Res_vs_Var([TRecNet_jetPretrain_data],par,'phi','pt',[90,510,20],save_loc='.',y_res='Residuals',LL_cuts=[-52,-50,-48])
        # Plot.Res_vs_Var([TRecNet_jetPretrain_data],par,'pt','phi',[-3,3,0.5],save_loc='.',y_res='Resolution',LL_cuts=[-52,-50,-48])


print("done :)")








