# Import useful packages
import uproot
import h5py
import json
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
import sys



# ---------- FUNCTION DEFINITIONS ---------- #



def readInData(reco_method,filename,eventnumbers):
    """
    Reads data into a dataframe from a file, and removes events that are not part of the test dataset (if list of test event numbers is given).

        Parameters:
            reco_method (str): Name of the reconstruction model (e.g.'KLFitter', 'TRecNet', etc.)
            filename (str): Name, including path, of root file to read data from.
            eventnumbers (list): Event numbers for the test dataset.

        Returns:
            df (pd.DataFrame): Dataframe full of data from the given file.
    """

    print('Reading in data for '+reco_method+' ...')

    # Open root file and its trees
    f = uproot.open(filename)

    # For TRecNet-based models ...
    if 'TRecNet' in reco_method and 'sys' not in reco_method:

        # Load the truth and reco data
        truth_arr = f['parton'].arrays()
        reco_arr = f['reco'].arrays()

        # Create dataframe with truth and reco data, as well as event numbers
        truth_df = ak.to_pandas(truth_arr)
        truth_df = truth_df.add_prefix('truth_')
        reco_df = ak.to_pandas(reco_arr)
        reco_df = reco_df.add_prefix('reco_')
        df = pd.concat([truth_df,reco_df], axis=1)
        df['eventNumber'] = truth_arr['eventNumber']

    # Systematics for TRecNet models
    elif 'TRecNet' in reco_method and 'sys' in reco_method:

        # Get whether it's up or down systematics
        sys_type = reco_method.split('_')[0]

        # Load sys data
        sys_arr = f['reco'].arrays()

        # ISSUE: this is cutting all events! Not sure why but shouldn't matter once it's all the new data? Idk, it really shouldnt be happening
        # Only take events with the same event numbers as the test data
        sel = np.isin(sys_arr['eventNumber'],eventnumbers)
        sys_arr = sys_arr[sel]


        # Create dataframe with reco data, as well as event numbers
        df = ak.to_pandas(sys_arr)
        #df = df.drop('index',axis=1) # index not in df I guess

        # Bit of a hack for now:
        df = df.add_prefix('reco_')
        df = df.rename(columns={'reco_eventNumber':'eventNumber'})

    # Systematics for non-TRecNet models
    elif 'TRecNet' not in reco_method and 'sys' in reco_method:

        # Get whether it's up or down systematics
        sys_type = reco_method.split('_')[0]

        # Load sys data
        sys_arr = f[sys_type].arrays()

        # ISSUE: this is cutting all events! Not sure why but shouldn't matter once it's all the new data? Idk, it really shouldnt be happening
        # Only take events with the same event numbers as the test data
        sel = np.isin(sys_arr['eventNumber'],eventnumbers)
        sys_arr = sys_arr[sel]


        # Create dataframe with reco data, as well as event numbers
        df = ak.to_pandas(sys_arr)
        #df = df.drop('index',axis=1) # index not in df I guess

        # Bit of a hack for now:
        df = df.add_prefix('reco_')
        df = df.rename(columns={'reco_eventNumber':'eventNumber'})




    # For likelihood-based models ... 
    else:

        # Load truth and reco data
        truth_arr = f['truth'].arrays()
        reco_arr = f['reco'].arrays()

        # ISSUE: this is cutting all events! Not sure why but shouldn't matter once it's all the new data? Idk, it really shouldnt be happening
        # Only take events with the same event numbers as the test data
        sel = np.isin(truth_arr['eventNumber'],eventnumbers)
        truth_arr = truth_arr[sel]
        reco_arr = reco_arr[sel]


        # Create dataframe with truth and reco data, as well as event numbers
        truth_df = ak.to_pandas(truth_arr)
        truth_df = truth_df.drop('index',axis=1)
        reco_df = ak.to_pandas(reco_arr)
        reco_df = reco_df.drop('index',axis=1)
        df = pd.concat([truth_df,reco_df], axis=1)

    

    return df





def Append_Calculations(df,model):
    """
        Calculate a few more observables that we might want to plot.

            Parameters:
                df (pd.DataFrame): Dataframe from which new observables will be calculated and to which they'll be appended.
                model (str): Name of the model for this dataframe.

            Returns:
                df (pd.DataFrame): Input dataframe with the new observables appended.
    """

    # Do so for both truth and reco
    for level in ['truth','reco']:

        # Skip doing this systematics ???? Bit confused about what this is ...
        if level=='truth' and 'sys' in model:
            continue

        # Append calculations for px and py for ttbar 
        df[level+'_ttbar_px'] = df[level+'_ttbar_pt']*np.cos(df[level+'_ttbar_phi'])
        df[level+'_ttbar_py'] = df[level+'_ttbar_pt']*np.sin(df[level+'_ttbar_phi'])

        # Chi2 doesn't have hadronic or leptonic tops, so we can't do the following for that model
        if model!='Chi2':

            # Append calculations for px and py for th and tl
            df[level+'_th_px'] = df[level+'_th_pt']*np.cos(df[level+'_th_phi'])
            df[level+'_th_py'] = df[level+'_th_pt']*np.sin(df[level+'_th_phi'])
            df[level+'_tl_px'] = df[level+'_tl_pt']*np.cos(df[level+'_tl_phi'])
            df[level+'_tl_py'] = df[level+'_tl_pt']*np.sin(df[level+'_tl_phi'])

            # Calculate ttbar deta and append
            th_vec = vector.arr({"pt": df[level+'_th_pt'], "phi": df[level+'_th_phi'], "eta": df[level+'_th_eta'],"mass": df[level+'_th_m']})
            tl_vec = vector.arr({"pt": df[level+'_tl_pt'], "phi": df[level+'_tl_phi'], "eta": df[level+'_tl_eta'],"mass": df[level+'_tl_m']}) 
            df[level+'_ttbar_deta'] = abs(th_vec.deltaeta(tl_vec))

            # If the model is KLFitter or PseudoTop ...
            if model=='KLFitter4' or model=='KLFitter6' or model=='PseudoTop':

                # Calculate and append dphi, Ht, yboost, and ystar for ttbar
                df[level+'_ttbar_dphi'] = abs(th_vec.deltaphi(tl_vec))
                df[level+'_ttbar_Ht'] = df[level+'_th_pt']+df[level+'_tl_pt']
                df[level+'_ttbar_yboost'] = 0.5*(df[level+'_th_y']+df[level+'_tl_y'])
                df[level+'_ttbar_ystar'] = 0.5*(df[level+'_th_y']-df[level+'_tl_y'])


    return df






def convertUnits(df_col,from_units,to_units):

    # Only bother converting if units are actually different
    if from_units!=to_units:

        # Figure out how many eV per the units we're converting from
        from_prefix = from_units[0] if from_units!='eV' else ''
        from_factor = 1 if from_prefix=='' else 10**6 if from_prefix=='M' else 10**9 if from_prefix=='G' else 10**12 if from_prefix=='T' else 0

        # Figure out how many eV per the units we're converting to
        to_prefix = to_units[0] if to_units!='eV' else ''
        to_factor = 1 if to_prefix=='' else 10**6 if to_prefix=='M' else 10**9 if to_prefix=='G' else 10**12 if to_prefix=='T' else 0

        # If we're not prepared to handle this unit, throw an error
        if from_factor==0 or to_factor==0:
            print('Units prefix not recognized (currently recognizes: eV, MeV, GeV or TeV.) Please use one of these, or edit this code.')
            sys.exit()

        # Check
        # print('Converting from '+from_units+' to '+to_units)
        # print('1 '+from_units+' = '+str(from_factor/to_factor)+to_units)

        # Convert from the previous units to the new units
        df_col = df_col*(from_factor/to_factor)


    return df_col



# CURRENTLY ONLY FOR ENERGY, INCLUDE OTHERS LATER???
def Scale_Data(df,truth_units,reco_units,par_specs):

    # Go through all the observables
    for (par,spec) in par_specs.items():
    	for obs in spec['observables']:
        
            # Look at observables with energy units
            if 'eV' in obs['unit']:

                # Convert from truth units to units specified for the observable
                if 'truth_'+par+'_'+obs['name'] in df.keys():
                    df['truth_'+par+'_'+obs['name']] = convertUnits(df['truth_'+par+'_'+obs['name']],truth_units,obs['unit'])

                # Convert from reco units to units specified for the observable
                if 'reco_'+par+'_'+obs['name'] in df.keys():
                    df['reco_'+par+'_'+obs['name']] = convertUnits(df['reco_'+par+'_'+obs['name']],reco_units,obs['unit'])

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

        


def unpack_observables(obs_specs):
    """
    Unpacks the observable definitions from the given dictionary (likely from a JSON file).

        Parameters:
            obs_specs (dict): Observable specifications, including name of the observables, alternative names, labels, units, and starting point, stopping point, and step size for the bins.

        Returns:
            observables (list of observable objects): Observables
    """

    # Create a list to store the observables in
    observables = []

    # For each observable specified ...
    for obs in obs_specs:

        # Get the ticks for the bins
        ticks = np.arange(obs['start'],obs['stop'],obs['step'])

        # Create the observable object and append it to the list
        observables.append(Observable(obs['name'],obs['label'],ticks,unit=obs['unit'],res=obs['res'],alt_names=obs['alt_names']))
    
    return observables



def unpack_particles(par_specs):
    """
    Unpacks the particle definitions from the given dictionary (likely from a JSON file).

        Parameters:
            par_specs (dict): Particle specifications, including name of the particles, alternative names, labels, and observables

        Returns:
            particles (list of particle objects): Particles
    """

    # Create a list to store the particles in
    particles = []

    # For each particle specified ...
    for par, specs in par_specs.items():

        # Get the observables for that particle
        observables = unpack_observables(specs['observables'])

        # Create the particle object and append it to the list
        particles.append(Particle(par,specs['label'],observables,alt_names=specs['alt_names']))

    return particles





def getCutDF(df,cut_on,max,min):
    """
    Creates a version of the input dataframe with the specified cuts (or lack thereof).

        Parameters:
            df (pd.DataFrame): Base dataframe.
            cut_on (str or None): Name of the thing which is being cut on, or None if no cut is desired.
            max (int or None): Maximum value, or None if there is no maximum desired..
            min (int or None): Minimum value, or None if there is no minimum desired.

        Returns:
            df_cut (pd.DataFrame): Dataframe with the implemented cuts, or the original dataframe if no cuts are desired.
    """

    # Keep data with quantity > min
    if max==None and min!=None:
        df_cut = df[df[cut_on]>min]

    # Keep data with quantity < max
    elif max!=None and min==None:
        df_cut = df[df[cut_on]<max]

    # Keep data with min < quantity < max
    elif max!=None and min!=None:
        df_cut = df[df[cut_on]>min]
        df_cut = df_cut[df_cut[cut_on]<max]

    # No cuts
    else:
        df_cut = df

    return df_cut






def makeAllDatasets(par_specs,model_specs,models_to_load,sys_to_load,datatype,eventnumbers):
    """
    Makes datasets objects for all of the models we'll want to plots.

        Parameters:
            model_specs (dict): Specifications for each of the models (likely read in from a JSON file).
            models_to_load (list of str): Names of the models for which dataset objects need to be created.
            sys_to_load (list of str): Names of the models for which systematics need to be included.
            datatype (str): Datatype (e.g. 'parton_e+mjets').
            eventnumbers (list): Event numbers for the test dataset.

        Returns:
            datasets (list of dataset objects): Dataset objects for each of the models specified.

    """

    # Create a list to hold all the datsets
    datasets = []

    # Going to want the total number of test events too
    total_test_events = len(eventnumbers)

    # Go through each model in the list of those we need to load
    for model_name in models_to_load:

        # Create a dataframe for that model, append some other variables we might want to plot, and scale data to desired units of energy
        df = readInData(model_name,model_specs[model_name]["input_file"],eventnumbers)
        df = Append_Calculations(df,model_name)
        print(df[['truth_th_m','reco_th_m']])
        df = Scale_Data(df,model_specs[model_name]["truth_units"],model_specs[model_name]["reco_units"],par_specs)
        print(df[['truth_th_m','reco_th_m']])


        # If we're going to need systematics for plots with this model ...
        if model_name in sys_to_load:

            # Create dataframe for up systematics
            up_df = readInData('sysUP_'+model_name,model_specs[model_name]["sysUP_input"],eventnumbers)
            up_df = Append_Calculations(df,model_name)
            up_df = Scale_Data(up_df,model_specs[model_name]["truth_units"],model_specs[model_name]["reco_units"],par_specs)

            # Create dataframe for down systematics
            down_df = readInData('sysDOWN_'+model_name,model_specs[model_name]["sysDOWN_input"],eventnumbers)
            down_df = Append_Calculations(df,model_name)
            down_df = Scale_Data(down_df,model_specs[model_name]["truth_units"],model_specs[model_name]["reco_units"],par_specs)

            # Create a dataset for each cut of the model we want to look at (is there a better way to do this, like within the plotting framework, so we don't have to make a bunch of dataframes?)
            for cut in model_specs[model_name]["cuts"]:

                # Grab the things we'll need
                cut_on, min, max, tag, color_scheme = cut["cut_on"], cut["min"], cut["max"], cut["tag"], cut["color_scheme"],

                # Get the dataframes with the specified cuts (or lackthereof)
                df_cut = getCutDF(df,cut_on,max,min)
                up_df_cut = getCutDF(up_df,cut_on,max,min)
                down_df_cut = getCutDF(down_df,cut_on,max,min)
                    
                # Create the dataset object and append it to our dictionary
                p = int(100*len(df_cut)/total_test_events)
                dataset = Dataset(df_cut,model_name,datatype,color_scheme,sysUP_df=up_df_cut,sysDOWN_df=down_df_cut,shortname=model_specs[model_name]["shortname"],cuts=tag,perc_events=p)
                datasets.append(dataset)
                    
        # If we don't need/want systematics ...
        else:

            # Create a dataset for each cut of the model we want to look at (is there a better way to do this, like within the plotting framework, so we don't have to make a bunch of dataframes?)
            for cut in model_specs[model_name]["cuts"]:

                # Grab the things we'll need
                cut_on, min, max, tag, color_scheme = cut["cut_on"], cut["min"], cut["max"], cut["tag"], cut["color_scheme"],

                # Get the dataframes with the specified cuts (or lackthereof)
                df_cut = getCutDF(df,cut_on,max,min)
                    
                # Create the dataset object and append it to our dictionary
                p = int(100*len(df_cut)/total_test_events)
                dataset = Dataset(df_cut,model_name,datatype,color_scheme,shortname=model_specs[model_name]["shortname"],cuts=tag,perc_events=p)
                datasets.append(dataset)

    return datasets

            
    


# ------------- MAIN CODE ------------- #


# Get the model specifications, plotting specifications, and observable specifications from the JSON file
plotting_info = json.load(open('plotting_info.json'))
model_specs = plotting_info['model_specs']
plot_specs = plotting_info['plot_specs']
par_specs = plotting_info['par_specs']

# Get the total number of test events and the list of test events
f_test = h5py.File(plotting_info['test_filename'],'r')
eventnumbers = np.array(f_test.get('eventNumber'))
f_test.close()

# Make particle definitions from the info in the JSON file
particles = unpack_particles(par_specs)

# Create lists of what models and systematics will need to be loaded
models_to_load = []
sys_to_load = []
for model_name in model_specs.keys():
    for plot, specs in plot_specs.items():
        if specs['models'][model_name]['plot']==True:
            models_to_load.append(model_name) if model_name not in models_to_load else models_to_load
        if specs['models'][model_name]['include_systematics']==True:
            sys_to_load.append(model_name) if model_name not in sys_to_load else sys_to_load
print('Preparing to load results for the following models: ',models_to_load)
print('Preparing to load systematics for the following models: ',sys_to_load)

# Make dataframes for models
loaded_models = makeAllDatasets(par_specs,model_specs,models_to_load,sys_to_load,plotting_info["datatype"],eventnumbers)
print("All data loaded.")



# Then later we plot (this should all be broken down more)

dir = 'full_plots/'    # This needs to probably be specified in json, and maybe we do something that (re)creates folders to put plots in

for par in particles:
    for var in par.observables:

        # Go through each of the models we loaded
        for dataset in loaded_models:

            # Check that we have this variable in our dataframe for this model
            if 'reco_'+par.name+'_'+var.name in dataset.df.keys():

                # If the json file says to plot TruthReco for that model, we do so
                TruthReco_Instructions = plot_specs['TruthReco']['models'][dataset.reco_method]
                if TruthReco_Instructions['plot']==True:
                    Plot.TruthReco_Hist(dataset,par,var,save_loc=dir+par.name+'/TruthReco/',nbins=TruthReco_Instructions['nbins'])


                # If the json file says to plot CM for that model, we do so
                CM_Instructions = plot_specs['CM']['models'][dataset.reco_method]
                if CM_Instructions['plot']==True:
                    for cut in CM_Instructions['pt_cuts']:
                        if cut=={}:
                            Plot.Confusion_Matrix(dataset,par,var,save_loc=dir+par.name+'/CM/')
                        elif len(cut)==1 and 'pt_low' in cut.keys():
                            Plot.Confusion_Matrix(dataset,par,var,save_loc=dir+par.name+'/CM/',pt_low=cut['pt_low'])
                        elif len(cut)==1 and 'pt_high' in cut.keys():
                            Plot.Confusion_Matrix(dataset,par,var,save_loc=dir+par.name+'/CM/',pt_high=cut['pt_high'])
                        else:
                            Plot.Confusion_Matrix(dataset,par,var,save_loc=dir+par.name+'/CM/',pt_low=cut['pt_low'],pt_high=cut['pt_high'])

            # If we can't plot this variable, skip it
            else:

                print(par.name+'_'+var.name+' was not reconstructed or calculated for '+dataset.reco_method+'. Skipping plot.')



        # Now for the resolution plots

        # Get the instructions for the resolution and res vs var plots
        Res_Instructions = plot_specs['Res']

        # Get the datasets for the models to be included in the resolution plots
        res_models = []
        for model in loaded_models:
            if Res_Instructions['models'][model.reco_method]['plot']==True and 'reco_'+par.name+'_'+var.name in model.df.keys():
                res_models.append(model)

        # Make sure we actually have something to plot!
        if res_models !=[]:

            # For each desired pt cut listed, create the resolution plot
            for cut in Res_Instructions['pt_cuts']:
                if cut=={}:
                    Plot.Res(res_models,par,var,save_loc=dir+par.name+'/Res/',nbins=Res_Instructions['nbins'],core_fit='nofit')
                elif len(cut)==1 and 'pt_low' in cut.keys():
                    Plot.Res(res_models,par,var,save_loc=dir+par.name+'/Res/',nbins=Res_Instructions['nbins'],core_fit='nofit',pt_low=cut['pt_low'])
                elif len(cut)==1 and 'pt_high' in cut.keys():
                    Plot.Res(res_models,par,var,save_loc=dir+par.name+'/Res/',nbins=Res_Instructions['nbins'],core_fit='nofit',pt_high=cut['pt_high'])
                else:
                    Plot.Res(res_models,par,var,save_loc=dir+par.name+'/Res/',nbins=Res_Instructions['nbins'],core_fit='nofit',pt_low=cut['pt_low'],pt_high=cut['pt_high'])




# Get the instructions for the res vs var plots
Res_vs_Var_Instructions = plot_specs['Res_vs_Var']





# Go through each particle
for par in particles:

    # For each of the res vs var plots we want to make for this particle
    for plot in Res_vs_Var_Instructions['plots'][par.name]:

        # Get the x and y observable objects
        for obs in par.observables:
            if plot['y_var']==obs.name: y_var = obs
            if plot['x_var']==obs.name: x_var = obs

        # Get the datasets for the models to be included in the resolution plots, which ALSO can reconstruct both the x and y variables
        res_vs_var_models = []
        for model in loaded_models:
            if Res_vs_Var_Instructions['models'][model.reco_method]['plot']==True and 'reco_'+par.name+'_'+y_var.name in model.df.keys() and 'reco_'+par.name+'_'+x_var.name in model.df.keys():
                res_vs_var_models.append(model)

        # Make the plot
        Plot.Res_vs_Var(res_vs_var_models,par,y_var,x_var,plot['xbins'],save_loc=dir+par.name+'/Res/',core_fit=Res_vs_Var_Instructions['core_fit'])
    

print('done! :)')



