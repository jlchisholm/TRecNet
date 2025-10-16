######################################################################
#                                                                    #
#  Plotter.py                                                        #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.16/25                                                #
#                                                                    #
#  Makes plots.                                                      #
#                                                                    #
######################################################################


# Import useful packages
import sys
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import uproot
import h5py
import json
import Util
from Particles_and_Observables import PARTICLES
from Variables_and_Datasets import Dataset, Variable
import Plots


class Plotter:
    """ 
    A class for making various kinds of plots.
    """
    
    def __init__(self, main_dir, f_dataset_config, plot_configs):
        """
        Initializes an Plotter object. 

            Parameters:
                main_dir (str): Directory (including path) to save plots in.
                particles (list of Particle objects): List of particles to be plotted.
                f_dataset_config (str): File name (and path) for the dataset config.
                plot_configs (dic): Dictionary of config files for each type of plot to be produced.
        """
        
        self.main_dir = 'plots/'+main_dir
        self.plots_to_make = plot_configs.keys()
        self.dataset_config = json.load(open(f_dataset_config))
        
        # Load config files for each plot type, and save the names of the reco models needed
        self.reco_models_to_plot = {}
        for plot_type, file_name in plot_configs.items():
            if plot_type=='TrainValLoss':
                self.train_val_loss_config = json.load(open(file_name))
                self.reco_models_to_plot[plot_type] = self.train_val_loss_config["TRecNet_models_to_plot"]
            elif plot_type=='TruthReco':
                self.truthreco_config = json.load(open(file_name))
                self.reco_models_to_plot[plot_type] = self.truthreco_config["reco_models_to_plot"]
            elif plot_type=='CM':
                self.cm_config = json.load(open(file_name))
                self.reco_models_to_plot[plot_type] = self.cm_config["reco_models_to_plot"]
            elif plot_type=='Res':
                self.res_config = json.load(open(file_name))
                self.reco_models_to_plot[plot_type] = self.res_config["reco_models_to_plot"]
            elif plot_type=='ResVsVar':
                self.res_vs_var_config = json.load(open(file_name))
                self.reco_models_to_plot[plot_type] = self.res_vs_var_config["reco_models_to_plot"]
            elif plot_type=='Sys':
                self.sys_config = json.load(open(file_name))
                self.reco_models_to_plot[plot_type] = self.sys_config["reco_models_to_plot"]
                            
        # Create "datasets"    
        self.datasets = {}
        for model, specs in self.dataset_config['Models'].items():
            if model in sum(self.reco_models_to_plot.values(),[]): # only make datasets for the models we're going to need
                avail_var_names= specs["available_variables"]
                for cut in specs["cuts"]: # separate dataset objects for separate cuts
                    
                    if cut["cut_on"]!=None:
                        with uproot.open(specs['nom_input']) as dataset_file: # Find the precentage of events in this cut (may remove this...)
                            cut_var = cut["cut_on"]
                            total_df = pd.DataFrame(dataset_file["reco"][cut_var],columns=[cut_var]) 
                            cut_df = self.getCutDF(total_df,cut_var,cut["max"],cut["min"])
                            perc_events = int(100*len(cut_df)/len(total_df))
                        self.datasets.update({model+'('+cut["tag"]+')': Dataset(model,cut["color_scheme"],avail_var_names,cut["tag"],cut_var,cut["max"],cut["min"],perc_events,specs["shortname"])})
                    else:
                        self.datasets.update({model+'('+cut["tag"]+')': Dataset(model,cut["color_scheme"],avail_var_names,reco_method_short=specs["shortname"])})        
            
            
    def getDatasetsToPlot(self, reco_models_to_plot):
        """
        Makes a list of the datasets we'll want to plot. This is different from reco models to plot, as datasets includes separate datasets for trimmed datasets (e.g. LL<-52 vs no cut)
        
            Parameters:
                reco_models_to_plot (list of str): List of reco model names.
            
            Return:
                datasets_to_plot (list of str): List of dataset names.
        """
        
        datasets_to_plot = []
        for name, dataset in self.datasets.items():  # go through all possible datasets
            if dataset.reco_method in reco_models_to_plot:
                datasets_to_plot.append(name)   # save the dataset if it's the right reco model
                
        return datasets_to_plot
            
            
    def getCutDF(self,df,cut_on,max_val,min_val):
        """
        Creates a version of the input dataframe with the specified cuts (or lack thereof).

            Parameters:
                df (pd.DataFrame): Base dataframe.
                cut_on (str or None): Name of the thing which is being cut on, or None if no cut is desired.
            
            Optional:
                max_val (int or None): Maximum value, or None if there is no maximum desired..
                min_val (int or None): Minimum value, or None if there is no minimum desired.

            Returns:
                df_cut (pd.DataFrame): Dataframe with the implemented cuts, or the original dataframe if no cuts are desired.
        """

        # Keep data with metric > min
        if max_val==None and min_val!=None:
            df_cut = df[df[cut_on]>min_val]

        # Keep data with metric < max
        elif max_val!=None and min_val==None:
            df_cut = df[df[cut_on]<max_val]

        # Keep data with min < metric < max
        elif max_val!=None and min_val!=None:
            df_cut = df[df[cut_on]>min_val]
            df_cut = df_cut[df_cut[cut_on]<max_val]

        # No cuts
        else:
            df_cut = df
            logger.warning('No cuts made.')

        return df_cut
    
    
    def getDataFrame(self,input_type,dataset_name,variable,extra_truth_vars=[],extra_reco_vars=[],cut_var=None):
        """
        Get truth and reco dataframes containing the given observable name. Also include any extra specified variables.
        
            Parameters:
                input_type (str): Type of data to put in the dataframe (i.e. 'nom', 'sysUP', or 'sysDOWN').
                dataset_name (str): Name of the dataset (e.g. 'KLFitter(LL>-52)').
                variable (Variable object): Variable of interest.
            
            Optional:
                extra_truth_vars (list of Variable objects): List of any additional truth variables that will be needed.
                extra_reco_vars (list of Variable objects): List of any additional reco variables that will be needed.
                cut_var (str): Name of the observable this dataset will be cut on (e.g. 'logLikelihood').
            
            Return:
                df (pd.DataFrame): Dataframe containing truth and reco values for the given observable. Also includes any extra truth and reco variables that were specified.
        """
        
        # Get the reco model object
        dataset = self.datasets[dataset_name] 
        
        # Set the variables (making sure we don't have observable name in there multiple times)
        truth_vars = [variable] + extra_truth_vars if variable not in extra_truth_vars else extra_truth_vars
        reco_vars = [variable] + extra_reco_vars if variable not in extra_reco_vars else extra_reco_vars
                
        # Read in data to pandas dataframe
        with uproot.open(self.dataset_config["Models"][dataset.reco_method][input_type+"_input"]) as data_file:
            
            # Nominal data
            if input_type=="nom":
                
                # Get truth data
                truth_df = data_file["parton"].arrays([v.name for v in truth_vars],library="pd") # only truth data for nominal events
                truth_df = truth_df.add_prefix('truth_')
                
                # Get reco data (including cut variable if necessary)
                reco_df = data_file["reco"].arrays([v.name for v in reco_vars],library="pd")
                reco_df = reco_df.add_prefix('reco_')
                if cut_var!=None: reco_df[cut_var] = data_file["reco"].arrays([cut_var],library="pd")
                
                # Make dataframe and ensure the units are correct
                df = pd.concat([truth_df,reco_df], axis=1)
                for var in list(set(truth_vars + reco_vars)):
                    df = Util.checkUnits(df,var.name,var.units)
                
            # Systematics data
            else:
                df = data_file[input_type].arrays([v.name for v in reco_vars],library="pd")
                df = df.add_prefix('reco_') 
                if cut_var!=None: df[cut_var] = data_file[input_type].arrays([cut_var],library="pd")
                for var in list(set(truth_vars + reco_vars)):
                    df = Util.checkUnits(df,variable.name,variable.units)
        
        return df
            

    
    def getDatasetList(self,variable,datasets_to_plot,extra_vars=[],with_systematics=False):
        """
        Get list of datasets for plots that require multiple datasets.
        
            Parameters:
                variable (Variable object): Variable of interest.
                datasets_to_plot (list of str): List of datasets to include.
                
            Optional:
                extra_vars (list of Variable objects): List of extra variables that may be needed (e.g. for cuts) (default: []).
                with_systematics (bool): Whether to include systematics in the datasets.
                
            Returns:
                datasets (list of Dataset objects): List of datasets.
        """
        
        # Iterate through each dataset (and save to list)
        datasets = []
        for dataset_name in datasets_to_plot:
            
            # Get the dataset object
            dataset = self.datasets[dataset_name] 
            
            # Get the variable of interest, or skip this dataset if it isn't available
            if variable.name not in dataset.avail_vars.keys():
                logger.info(variable.name+' is not an available variable for '+dataset.reco_method+'. Skipping this plot.')
                continue
                    
            # Get the dataframe and make cut if necessary
            if dataset.cut_var!=None:
                df = self.getDataFrame('nom',dataset_name,variable,extra_truth_vars=extra_vars,extra_reco_vars=extra_vars,cut_var=dataset.cut_var)
                df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)
            else:
                df = self.getDataFrame('nom',dataset_name,variable,extra_truth_vars=extra_vars,extra_reco_vars=extra_vars)

            # Link data to the dataset
            dataset.link_temp_df(df)
            
            # Also get systematics dataframes if required
            if with_systematics:
                up_df = self.getDataFrame('sysUP',dataset_name,variable,extra_truth_vars=extra_vars,extra_reco_vars=extra_vars,cut_var=dataset.cut_var)
                down_df = self.getDataFrame('sysDOWN',dataset_name,variable,extra_truth_vars=extra_vars,extra_reco_vars=extra_vars,cut_var=dataset.cut_var)
                if dataset.cut_var!=None:
                    up_df = self.getCutDF(up_df,dataset.cut_var,dataset.cut_max,dataset.cut_min)
                    down_df = self.getCutDF(down_df,dataset.cut_var,dataset.cut_max,dataset.cut_min)
                dataset.link_temp_sysUP_df(up_df)
                dataset.link_temp_sysDOWN_df(down_df)

            # Save to list
            datasets.append(dataset)
            
        return datasets
    
    
    def makeTrainValLossPlots(self):
        """
        Make training/validation loss plots.
        """
        
        # Get the plotting instructions
        loss_metric = self.train_val_loss_config["loss_metric"]
        extra_metrics = self.train_val_loss_config["extra_metrics"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(self.reco_models_to_plot['TrainValLoss'])
        
        # Iterate through each dataset (and save to list)
        datasets = []
        for dataset_name in datasets_to_plot:
            
            # Get the dataset object
            dataset = self.datasets[dataset_name] 
            
            # Open the training history and link it to the dataset
            train_history = np.load(self.dataset_config['Models'][dataset.reco_method]["train_history"],allow_pickle=True).item()
            dataset.link_train_history(train_history)

            # Save to list
            datasets.append(dataset)
            
        # Make the plot!
        Plots.TrainValLoss_Plot(datasets,loss_metric,self.main_dir+'/TrainValLoss/',extra_metrics)
        print('TrainValLoss plots completed.')  
                 
               
    def makeTruthRecoPlots(self):
        """
        Makes truth reco plots.
        """
        
        # Get the plotting instructions
        observables_to_plot = self.truthreco_config["variables"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(self.reco_models_to_plot['TruthReco'])

        # Iterate through the observables
        for par, obs in observables_to_plot.items():
            for ob, specs in obs.items():
                
                # Read the specs
                x_min = specs["min"]
                x_max = specs["max"]
                nbins = specs["nbins"]
                    
                # Iterate through each dataset
                for dataset_name in datasets_to_plot:
                    
                    # Get the dataset object
                    dataset = self.datasets[dataset_name] 
                    
                    # Get the variable of interest, or skip this dataset if it isn't available
                    if par+'_'+ob in dataset.avail_vars.keys():
                        variable = dataset.avail_vars[par+'_'+ob]
                    else:
                        logger.info(par+'_'+ob+' is not an available variable for '+dataset.reco_method+'. Skipping this plot.')
                        continue
                    
                    # Get dataframe and make cut if necessary
                    if dataset.cut_var!=None:
                        df = self.getDataFrame('nom',dataset_name,variable,cut_var=dataset.cut_var)
                        df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)
                    else:
                        df = self.getDataFrame('nom',dataset_name,variable)

                    # Link data to the dataset
                    dataset.link_temp_df(df)
            
                    # Make the plot
                    Plots.TruthReco_Hist(dataset,variable,x_min,x_max,nbins,self.main_dir+'/'+par+'/TruthReco/')

        print('TruthReco plots completed.')  
        
        
    def makeCMPlots(self):
        """
        Make confusion matrix plots.
        """
        
        # Get the plotting instructions
        observables_to_plot = self.cm_config["variables"]
        even_stats = self.cm_config["even_stats_binning"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(self.reco_models_to_plot['CM'])
            
        # Iterate through the observables
        for par, obs in observables_to_plot.items():
            for ob, specs in obs.items():
                
                # Construct the variable of interest
                variable = Variable(PARTICLES[par],PARTICLES[par].get_observable(ob))
                
                # Read the specs and get ticks
                if even_stats:
                    nbins = specs["even_stats_bins"]["nbins"]
                    folded_bins = specs["even_stats_bins"]["folded_bins"]
                    with h5py.File(self.dataset_config['Test_Data']['nom_input'],'r') as test_file:
                        temp_df = pd.DataFrame(np.array(test_file.get(variable.name)),columns=[variable.name])
                    ticks, tick_labels = Util.getEvenStatsTicks(temp_df[variable.name],variable.observable,folded_bins,nbins)
                    stats_tag = '(stats_binning)'
                else:
                    x_min = specs["custom_bins"]["min"]
                    x_max = specs["custom_bins"]["max"]
                    step = specs["custom_bins"]["step"]
                    folded_bins = specs["custom_bins"]["folded_bins"]
                    ticks, tick_labels = Util.getTicks(variable.observable,x_min,x_max,step,folded_bins)
                    stats_tag = ''
                
                # Iterate through each dataset
                for dataset_name in datasets_to_plot:
                    
                    # Get the dataset object
                    dataset = self.datasets[dataset_name] 
                    
                    # Get the variable of interest, or skip this dataset if it isn't available
                    if variable.name not in dataset.avail_vars.keys():
                        logger.info(variable.name+' is not an available variable for '+dataset.reco_method+'. Skipping this plot.')
                        continue
                        
                    # Get dataframe and make cut if necessary
                    if dataset.cut_var!=None:
                        df = self.getDataFrame('nom',dataset_name,variable,cut_var=dataset.cut_var)
                        df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)
                    else:
                        df = self.getDataFrame('nom',dataset_name,variable)

                    # Link data to the dataset
                    dataset.link_temp_df(df)
                    
                    # Make the plot
                    Plots.Confusion_Matrix(dataset,variable,ticks,tick_labels,folded_bins=folded_bins,tag=stats_tag,save_loc=self.main_dir+'/'+par+'/CM/')
                    
        print('CM plots completed.')  
        
        
    def makeResPlots(self):
        """
        Make resolution/residuals plots.
        """
        
        # Get the plotting instructions
        observables_to_plot = self.res_config["variables"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(self.reco_models_to_plot['Res'])
        
        # Iterate through the observables
        for par, obs in observables_to_plot.items():
            for ob, specs in obs.items():
                
                # Construct the variable of interest
                variable = Variable(PARTICLES[par],PARTICLES[par].get_observable(ob))
                
                # Get datasets (adding pt as an extra variable if we're going to cut on it but don't already have it)
                if (len(specs["pt_cuts"])==1 and specs["pt_cuts"][0]=={}) or variable.observable.name=='pt':
                    datasets = self.getDatasetList(variable,datasets_to_plot)
                else:
                    pt_var = Variable(PARTICLES[par],PARTICLES[par].get_observable('pt'))
                    datasets = self.getDatasetList(variable,datasets_to_plot,extra_vars=[pt_var])
                
                # Iterate through each of the pt cuts
                for pt_cut in specs["pt_cuts"]:
                    
                    # Need a new list of cut datasets
                    cut_datasets = []
                    for dataset in datasets:
                        
                        # Cut to specified pt range, if desired
                        cut_df = dataset.df.copy()
                        if pt_cut!={}:
                            if "pt_low" in pt_cut.keys():
                                cut_df = cut_df[cut_df['truth_'+par+'_pt']>pt_cut["pt_low"]]
                            if "pt_high" in pt_cut.keys():
                                cut_df = cut_df[cut_df['truth_'+par+'_pt']<pt_cut["pt_high"]]
                                
                        # Save new cut dataset
                        cut_dataset = dataset.copy()
                        cut_dataset.link_temp_df(cut_df)
                        cut_datasets.append(cut_dataset)
                        
                    # Creating pt tag for later saving/naming
                    if pt_cut=={}:
                        pt_tag = ''
                    elif len(pt_cut)==1 and 'pt_low' in pt_cut.keys():
                        pt_tag = '(p_T>'+str(pt_cut["pt_low"])+')'
                    elif len(pt_cut)==1 and 'pt_high' in pt_cut.keys():
                        pt_tag = '(p_T<'+str(pt_cut["pt_high"])+')'
                    else:
                        pt_tag = '('+str(pt_cut["pt_low"])+'<p_T<'+str(pt_cut["pt_high"])+')'

                    # Now plot all these datasets together!
                    Plots.Res_Hist(cut_datasets,variable,save_loc=self.main_dir+'/'+par+'/Res/',tag=pt_tag,nbins=specs["nbins"],include_moments=specs["include_moments"])
                
        print('Res plots completed.')  
        
        
        
    def makeResVsVarPlots(self):
        """
        Make residual/resolution vs. variable plots.
        """
        
        # Get the plotting instructions
        observables_to_plot = self.res_vs_var_config["variables"]
        even_stats = self.res_vs_var_config["even_stats_binning"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(self.reco_models_to_plot['ResVsVar'])
                
        # Iterate through the observables
        for par, plot_requests in observables_to_plot.items():
            for plot_specs in plot_requests:
                
                # Get some important particle observable info
                x_ob = plot_specs["x_obs"]
                y_ob = plot_specs["y_obs"]
                folded_bins = plot_specs["folded_bins"]
                
                # Construct the variables of interest
                x_variable = Variable(PARTICLES[par],PARTICLES[par].get_observable(x_ob))
                y_variable = Variable(PARTICLES[par],PARTICLES[par].get_observable(y_ob))
                
                # Get datasets
                extra_vars = [y_variable] if y_ob!=x_ob else []
                datasets = self.getDatasetList(x_variable,datasets_to_plot,extra_vars=extra_vars)
                
                # Read the specs and get ticks
                if even_stats:
                    nbins = plot_specs["n_even_stats_bins"]
                    with h5py.File(self.dataset_config['Test_Data']['nom_input'],'r') as test_file:
                        temp_df = pd.DataFrame(np.array(test_file.get(x_variable.name)),columns=[x_variable.name])
                    ticks, tick_labels = Util.getEvenStatsTicks(temp_df[x_variable.name],x_variable.observable,folded_bins,nbins)
                    stats_tag = '(stats_binning)'
                else:
                    x_min = plot_specs["custom_bins"][0]
                    x_max = plot_specs["custom_bins"][1]
                    step = plot_specs["custom_bins"][2]
                    ticks, tick_labels = Util.getTicks(x_variable.observable,x_min,x_max,step,folded_bins)
                    stats_tag = ''

                # Make plot!
                Plots.Res_vs_Var(datasets,y_variable,x_variable,ticks,tick_labels,folded_bins=folded_bins,save_loc=self.main_dir+par+'/ResVsVar/',tag=stats_tag)
                
        print('ResVsVar plots completed.') 
        
        
        
    def makeSysPlots(self):
        """
        Makes systematics plots.
        """
        
        # Get the plotting instructions
        observables_to_plot = self.sys_config["variables"]
        even_stats = self.sys_config["even_stats_binning"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(self.reco_models_to_plot['Sys'])
        
        # Iterate through the observables
        for par, obs in observables_to_plot.items():
            for ob, specs in obs.items():
                
                # Construct the variable of interest
                variable = Variable(PARTICLES[par],PARTICLES[par].get_observable(ob))
                
                # Get datasets
                datasets = self.getDatasetList(variable,datasets_to_plot,with_systematics=True)
                
                # Read the specs and get ticks
                if even_stats:
                    nbins = specs["even_stats_bins"]["nbins"]
                    folded_bins = specs["even_stats_bins"]["folded_bins"]
                    with h5py.File(self.dataset_config['Test_Data']['nom_input'],'r') as test_file:
                        temp_df = pd.DataFrame(np.array(test_file.get(variable.name)),columns=[variable.name])
                    ticks, tick_labels = Util.getEvenStatsTicks(temp_df[variable.name],variable.observable,folded_bins,nbins)
                    stats_tag = '(stats_binning)'
                else:
                    x_min = specs["custom_bins"]["min"]
                    x_max = specs["custom_bins"]["max"]
                    step = specs["custom_bins"]["step"]
                    folded_bins = specs["custom_bins"]["folded_bins"]
                    ticks, tick_labels = Util.getTicks(variable.observable,x_min,x_max,step,folded_bins)
                    stats_tag = ''
                
                # Make plot!
                Plots.Sys_Hist(datasets,variable,ticks,tick_labels,folded_bins=folded_bins,save_loc=self.main_dir+par+'/Sys/',tag=stats_tag)
                    
                
    def makePlots(self):
        """
        Makes all desired plots.
        """
        
        if 'TrainValLoss' in self.plots_to_make:
            self.makeTrainValLossPlots()
        if 'TruthReco' in self.plots_to_make:
            self.makeTruthRecoPlots()
        if 'CM' in self.plots_to_make:
            self.makeCMPlots()
        if 'Res' in self.plots_to_make:
            self.makeResPlots()    
        if 'ResVsVar' in self.plots_to_make:
            self.makeResVsVarPlots()
        if 'Sys' in self.plots_to_make:
            self.makeSysPlots()    
        
        
        print('All plots completed.')    
    
        
        