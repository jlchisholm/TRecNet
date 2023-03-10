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
import os, sys
from argparse import ArgumentParser




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

    # When loading systematics ...
    if 'sys' in reco_method:

        # Get whether it's up or down systematics
        sys_type = reco_method.split('_')[0]

        # Slightly different names for neural network and other models -- something that could be fixed later
        if 'TRecNet' in reco_method:

            # Load systematics data
            sys_arr = f['reco'].arrays()

            # Create dataframe with systematics data, as well as event numbers, jet number, and log likelihood
            # And rename such that the naming is the same that it is in the other systematics datasets
            # Hold up hold up -- log likelihood???
            df = ak.to_pandas(sys_arr)
            df = df.add_prefix(sys_type+'_')   # Want to add sysUP or sysDOWN prefix to match the other files
            df = df.rename(columns={sys_type+'_eventNumber':'eventNumber',sys_type+'_jet_n':'jet_n',sys_type+'_logLikelihood':'logLikelihood'})

        else:

            # Load sys data and create dataframe with systematics data
            sys_arr = f[sys_type].arrays()
            df = ak.to_pandas(sys_arr)

    # For non systematics things ...
    else:

        # Slightly different names for neural network and other models -- something that could be fixed later
        if 'TRecNet' in reco_method:

            # Load the truth and reco data
            truth_arr = f['parton'].arrays()
            reco_arr = f['reco'].arrays()

            # Create dataframe with truth and reco data, as well as event numbers
            truth_df = ak.to_pandas(truth_arr)
            truth_df = truth_df.add_prefix('truth_')
            truth_df = truth_df.rename(columns={'truth_eventNumber':'eventNumber'})
            reco_df = ak.to_pandas(reco_arr)
            reco_df = reco_df.add_prefix('reco_')
            df = pd.concat([truth_df,reco_df], axis=1)

        else:

            # Load truth and reco data
            truth_arr = f['truth'].arrays()
            reco_arr = f['reco'].arrays()

            # Only take events with the same event numbers as the test data, for a more apples to apples comparison
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


def appendCalculations(df,model_name):
    """
        Calculate a few more observables that we might want to plot.

            Parameters:
                df (pd.DataFrame): Dataframe from which new observables will be calculated and to which they'll be appended.
                model_name (str): Name of the model for this dataframe.

            Returns:
                df (pd.DataFrame): Input dataframe with the new observables appended.
    """

    # Do so for both truth and reco, or sysUP or sysDOWN, depending on the dataframe
    levels = ['truth','reco'] if 'sys' not in model_name else [model_name.split('_')[0]]
    for level in levels:

        # Append calculations for px and py for ttbar 
        df[level+'_ttbar_px'] = df[level+'_ttbar_pt']*np.cos(df[level+'_ttbar_phi'])
        df[level+'_ttbar_py'] = df[level+'_ttbar_pt']*np.sin(df[level+'_ttbar_phi'])

        # Chi2 doesn't have hadronic or leptonic tops, so we can't do the following for that model
        if 'Chi2' not in model_name:

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
            if 'TRecNet' not in model_name:

                # Calculate and append dphi, Ht, yboost, and ystar for ttbar
                df[level+'_ttbar_dphi'] = abs(th_vec.deltaphi(tl_vec))
                df[level+'_ttbar_Ht'] = df[level+'_th_pt']+df[level+'_tl_pt']
                df[level+'_ttbar_yboost'] = 0.5*(df[level+'_th_y']+df[level+'_tl_y'])
                df[level+'_ttbar_ystar'] = 0.5*(df[level+'_th_y']-df[level+'_tl_y'])


    return df


def convertUnits(df_col,from_units,to_units):
    """
    Converts the data in <df_col> from <from_units> to <to_units>.
    Note: Can currently only handle particular energy units.

        Parameters:
            df_col (pd.DataFrame column): Observable data to be converted.
            from_units (str): Units the data is currently in (can currently handle 'eV', 'MeV', 'GeV', and 'TeV').
            to_units (str): Units to the convert the data into (can currently handle 'eV', 'MeV', 'GeV', and 'TeV').

        Returns:
            df_col (pd.DataFrame column): Observable data after the unit conversion.

    """

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


def scaleData(df,truth_units,reco_units,par_specs):
    """
    Scales all the data so everything is in the same units, specified in the particle specifications read in from the JSON file.
    Note: Can currently only handle particular energy units.

        Parameters:
            df (pd.DataFrame): Data to be scaled.
            truth_units (str): Units used in the truth data for energies.
            reco_units (str): Units used in the reco data for energies.
            par_specs (dict): Particle specifications, read in from the JSON file.

        Returns:
            df (pd.DataFrame): Data, after scaling.
    """

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


def unpackObservables(obs_specs):
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


def unpackParticles(par_specs):
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
        observables = unpackObservables(specs['observables'])

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
        df = appendCalculations(df,model_name)
        df = scaleData(df,model_specs[model_name]["truth_units"],model_specs[model_name]["reco_units"],par_specs)


        # If we're going to need systematics for plots with this model ...
        if model_name in sys_to_load:

            # Create dataframe for up systematics
            up_df = readInData('sysUP_'+model_name,model_specs[model_name]["sysUP_input"],eventnumbers)
            up_df = appendCalculations(up_df,'sysUP_'+model_name)
            up_df = scaleData(up_df,model_specs[model_name]["truth_units"],model_specs[model_name]["reco_units"],par_specs)

            # Create dataframe for down systematics
            down_df = readInData('sysDOWN_'+model_name,model_specs[model_name]["sysDOWN_input"],eventnumbers)
            down_df = appendCalculations(down_df,'sysDOWN_'+model_name)
            down_df = scaleData(down_df,model_specs[model_name]["truth_units"],model_specs[model_name]["reco_units"],par_specs)

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


def createDirectories(main_dir):
    """
    Creates directories to store the plots in, if they do not already exist.

        Parameters: 
            main_dir (str): Name (including path) of the primary directory you want to save the plots in.

        Returns:
            Creates all the <main_dir>, with directories 'th/', 'tl/', and 'ttbar/' within, and directories 'TruthReco/', 'Sys/', 'CM/', and 'Res/' within each of the three aforementioned directories.
    """
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    for par_dir in ['th/','tl/','ttbar/']:
        if not os.path.exists(main_dir+par_dir):
            os.mkdir(main_dir+par_dir) 
    for plot_dir in ['TruthReco/','Sys/','CM/','Res/']:
        for par_dir in ['th/','tl/','ttbar/']:
            if not os.path.exists(main_dir+par_dir+plot_dir):
                os.mkdir(main_dir+par_dir+plot_dir)


def getLoadLists(plotting_instr):
    """
    Gets lists of the models and systematics we'll need to load to make the desired plots.

        Parameters:
            plotting_instr (dict): Dictionary containing the types of plots we want to make and their specifications (from the JSON file).

        Returns:
            models_to_load (list of str): Names of the models we'll need data from.
            sys_to_load (list of str): Names of the models we'll need systematics from.
    """

    models_to_load = []
    sys_to_load = []
    for plot_type, instr in plotting_instr.items():
        if plot_type=='TruthReco' or plot_type=='CM':
            for model_name, model_instr in instr['models'].items():
                if model_instr['plot']:
                    models_to_load.append(model_name) if model_name not in models_to_load else models_to_load
        elif plot_type=='Sys':
            for model_name, plot_model in instr['include_model'].items():
                if plot_model:
                    models_to_load.append(model_name) if model_name not in models_to_load else models_to_load
                    sys_to_load.append(model_name) if model_name not in sys_to_load else sys_to_load
        else:
            for model_name, plot_model in instr['include_model'].items():
                if plot_model:
                    models_to_load.append(model_name) if model_name not in models_to_load else models_to_load

    print('Preparing to load results for the following models: ',models_to_load)
    print('Preparing to load systematics for the following models: ',sys_to_load)    

    return models_to_load, sys_to_load


def makeTruthRecoPlots(loaded_models, TruthReco_Instructions, par, var, main_dir):
    """
    Makes all the desired truth-reco plots.

        Parameters:
            loaded_models (list of dataset objects): Datasets that are loaded and available to use in plots.
            TruthReco_Instructions (dict): Specifications for making the truth-reco histograms (from JSON file).
            par (particle object): Particle to be plotted.
            var (observable object): Observable to be plotted.
            main_dir (str): Main directory where all plots will be saved.

        Returns:
            Saves several truth-reco histograms as images in <main_dir>.
    """

    truthreco_models = [model for model in loaded_models if TruthReco_Instructions['models'][model.reco_method]['plot'] and 'reco_'+par.name+'_'+var.name in model.df.keys()]

    if truthreco_models != []:
        for dataset in truthreco_models:
            Plot.TruthReco_Hist(dataset,par,var,save_loc=main_dir+par.name+'/TruthReco/',nbins=plotting_instr['TruthReco']['models'][dataset.reco_method]['nbins'])
    else:
        print('Please select models if you want truth-reco plots to be produced.')


def makeCMPlots(loaded_models, CM_Instructions, par, var, main_dir):
    """
    Makes all the desired confusion matrix plots.

        Parameters:
            loaded_models (list of dataset objects): Datasets that are loaded and available to use in plots.
            CM_Instructions (dict): Specifications for making the confusion matrix plots (from JSON file).
            par (particle object): Particle to be plotted.
            var (observable object): Observable to be plotted.
            main_dir (str): Main directory where all plots will be saved.

        Returns:
            Saves several confusion matrix plots as images in <main_dir>.
    """

    cm_models = [model for model in loaded_models if CM_Instructions['models'][model.reco_method]['plot'] and 'reco_'+par.name+'_'+var.name in model.df.keys()]

    if cm_models != []:
        for dataset in cm_models:
            for cut in CM_Instructions['models'][dataset.reco_method]['pt_cuts']:
                if cut=={}:
                    Plot.Confusion_Matrix(dataset,par,var,save_loc=main_dir+par.name+'/CM/')
                elif len(cut)==1 and 'pt_low' in cut.keys():
                    Plot.Confusion_Matrix(dataset,par,var,save_loc=main_dir+par.name+'/CM/',pt_low=cut['pt_low'])
                elif len(cut)==1 and 'pt_high' in cut.keys():
                    Plot.Confusion_Matrix(dataset,par,var,save_loc=main_dir+par.name+'/CM/',pt_high=cut['pt_high'])
                else:
                    Plot.Confusion_Matrix(dataset,par,var,save_loc=main_dir+par.name+'/CM/',pt_low=cut['pt_low'],pt_high=cut['pt_high'])
    else: 
        print('Please select models if you want confusion matrix plots to be produced.')


def makeSysPlots(loaded_models, Sys_Instructions, par, var, main_dir):

    sys_models = [model for model in loaded_models if Sys_Instructions['include_model'][model.reco_method] and 'reco_'+par.name+'_'+var.name in model.df.keys()]

    if sys_models != []:
        Plot.Sys_Hist(sys_models,par,var,save_loc=main_dir+par.name+'/Sys/')

    else:
        print('Please select models if you want systematics plots to be produced.')


def makeResPlots(loaded_models, Res_Instructions, par, var, main_dir):
    """
    Makes all the desired resolution (or residual) plots.

        Parameters:
            loaded_models (list of dataset objects): Datasets that are loaded and available to use in plots.
            Res_Instructions (dict): Specifications for making the resolution (or residual) histograms (from JSON file).
            par (particle object): Particle to be plotted.
            var (observable object): Observable to be plotted.
            main_dir (str): Main directory where all plots will be saved.

        Returns:
            Saves several resolution (or residual) histograms as images in <main_dir>.
    """


    # Get the datasets for the models to be included in the resolution plots
    res_models = [model for model in loaded_models if Res_Instructions['include_model'][model.reco_method] and 'reco_'+par.name+'_'+var.name in model.df.keys()]

    # Make sure we actually have something to plot!
    if res_models !=[]:

        # For each desired pt cut listed, create the resolution plot
        for cut in Res_Instructions['pt_cuts']:
            if cut=={}:
                Plot.Res_Hist(res_models,par,var,save_loc=main_dir+par.name+'/Res/',nbins=Res_Instructions['nbins'],core_fit=Res_Instructions['core_fit'],include_moments=Res_Instructions['include_moments'])
            elif len(cut)==1 and 'pt_low' in cut.keys():
                Plot.Res_Hist(res_models,par,var,save_loc=main_dir+par.name+'/Res/',nbins=Res_Instructions['nbins'],core_fit=Res_Instructions['core_fit'],include_moments=Res_Instructions['include_moments'],pt_low=cut['pt_low'])
            elif len(cut)==1 and 'pt_high' in cut.keys():
                Plot.Res_Hist(res_models,par,var,save_loc=main_dir+par.name+'/Res/',nbins=Res_Instructions['nbins'],core_fit=Res_Instructions['core_fit'],include_moments=Res_Instructions['include_moments'],pt_high=cut['pt_high'])
            else:
                Plot.Res_Hist(res_models,par,var,save_loc=main_dir+par.name+'/Res/',nbins=Res_Instructions['nbins'],core_fit=Res_Instructions['core_fit'],include_moments=Res_Instructions['include_moments'],pt_low=cut['pt_low'],pt_high=cut['pt_high'])
    else:
        print('Please select models if you want resolution (or residual) plots to be produced.')


def makeResVsVarPlots(loaded_models, Res_vs_Var_Instructions, par, main_dir):
    """
    Makes all the desired resolution (or residual) vs variable plots.

        Parameters:
            loaded_models (list of dataset objects): Datasets that are loaded and available to use in plots.
            Res_vs_Var_Instructions (dict): Specifications for making the resolution (or residual) vs variable plots (from JSON file).
            par (particle object): Particle to be plotted.
            var (observable object): Observable to be plotted.
            main_dir (str): Main directory where all plots will be saved.

        Returns:
            Saves several resolution (or residual) vs variable plots as images in <main_dir>.
    """


    # For each of the res vs var plots we want to make for this particle
    for plot in Res_vs_Var_Instructions['plots'][par.name]:

        # Get the x and y observable objects
        for obs in par.observables:
            if plot['y_var']==obs.name: y_var = obs
            if plot['x_var']==obs.name: x_var = obs

        # Get the datasets for the models to be included in the resolution plots, which ALSO can reconstruct both the x and y variables
        res_vs_var_models = []
        for model in loaded_models:
            if Res_vs_Var_Instructions['include_model'][model.reco_method] and 'reco_'+par.name+'_'+y_var.name in model.df.keys() and 'reco_'+par.name+'_'+x_var.name in model.df.keys():
                res_vs_var_models.append(model)

        # Make the plot
        if res_vs_var_models != []:
            Plot.Res_vs_Var(res_vs_var_models,par,y_var,x_var,plot['xbins'],save_loc=main_dir+par.name+'/Res/',core_fit=Res_vs_Var_Instructions['core_fit'])
        else:
            print('Please select models if you want resolution (or residual) vs variable plots to be produced.')

        
def makePlots(loaded_models, plotting_instr, particles, main_dir):
    """
    Makes all the desired plots.

        Parameters:
            loaded_models (list of dataset objects): Datasets that are loaded and available to use in plots.
            plotting_instr (dict): Plotting specifications (from the JSON file) for the plot types that we actually want to make.
            particles (list of particle objects): Particles to be plotted.
            main_dir (str): Main directory where all plots will be saved.

        Returns:
            Saves several plots as images in <main_dir>.
    """

    for par in particles:

        # For these plots, we'll just plot all the observables (for now anyways) instead of specifying them in the json file
        if 'TruthReco' in plotting_instr or 'CM' in plotting_instr or 'Res' in plotting_instr or 'Sys' in plotting_instr:
            for var in par.observables:
                if 'TruthReco' in plotting_instr:
                    makeTruthRecoPlots(loaded_models,plotting_instr['TruthReco'],par,var, main_dir)
                if 'CM' in plotting_instr:
                    makeCMPlots(loaded_models, plotting_instr['CM'], par, var, main_dir)  
                if 'Sys' in plotting_instr:
                    makeSysPlots(loaded_models, plotting_instr['Sys'], par, var, main_dir)     
                if 'Res' in plotting_instr:
                    makeResPlots(loaded_models, plotting_instr['Res'], par, var, main_dir)

                    
        # Res vs Var plots don't go through each observable -- only some variables are plotted against eachother (specified in json file)
        if 'Res_vs_Var' in plotting_instr:
            makeResVsVarPlots(loaded_models, plotting_instr['Res_vs_Var'],par, main_dir)






# ------------- MAIN CODE ------------- #


# Get JSON file name of plotting info from command line and load the file
parser = ArgumentParser()
parser.add_argument('--plotting_info', help='JSON file name (including path) that contains the plotting specifications you wish to use.', type=str, required=True)
args = parser.parse_args()
f_plotting_info = args.plotting_info
plotting_info = json.load(open(f_plotting_info))

# Create directories for plots if they don't already exist
main_dir = plotting_info['save_loc']+'plots/'
createDirectories(main_dir)

# Get the model specifications, plotting specifications, and observable specifications from the JSON file
model_specs = plotting_info['model_specs']
plot_specs = plotting_info['plot_specs']
par_specs = plotting_info['par_specs']

# Get the total number of test events and the list of test events
f_test = h5py.File(plotting_info['test_filename'],'r')
eventnumbers = np.array(f_test.get('eventNumber'))
f_test.close()

# Make particle definitions from the info in the JSON file
particles = unpackParticles(par_specs)

# Get plotting instructions for the things we actually want to plot
plotting_instr = {}
for plot, specs in plot_specs.items():
    if specs['makePlots']:
        plotting_instr[plot] = plot_specs[plot]
if plotting_instr == {}:
    print('Please select something to be plotted.')
    sys.exit()

# Create lists of what models and systematics will need to be loaded
models_to_load,sys_to_load = getLoadLists(plotting_instr)
if models_to_load == []:
    print('Please select models to be plotted.')
    sys.exit()

# Make dataframes for models
loaded_models = makeAllDatasets(par_specs,model_specs,models_to_load,sys_to_load,plotting_info["datatype"],eventnumbers)
print("All data loaded.")

# Make the desired plots
makePlots(loaded_models, plotting_instr,particles, main_dir)

print('done! :)')



