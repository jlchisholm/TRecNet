######################################################################
#                                                                    #
#  Plotter.py                                                        #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.10/25                                                 #
#                                                                    #
#  Makes plots.                                                      #
#                                                                    #
######################################################################


# Import useful packages
import numpy as np
import pandas as pd
import uproot
import h5py
import json
import Util
from ParticleObservables import PARTICLES
from DataStructures import Dataset
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
        
        self.main_dir = main_dir
        self.plots_to_make = plot_configs.keys()
        self.dataset_config = json.load(open(f_dataset_config))
        
        # Load config files for each plot type
        for plot_type, file_name in plot_configs.items():
            if plot_type=='TruthReco':
                self.truthreco_config = json.load(open(file_name))
            elif plot_type=='CM':
                self.cm_config = json.load(open(file_name))
            elif plot_type=='Res':
                self.res_config = json.load(open(file_name))
            elif plot_type=='ResVsVar':
                self.res_vs_var_config = json.load(open(file_name))
            elif plot_type=='Sys':
                self.sys_config = json.load(open(file_name))
            
        # Create "datasets"    
        self.datasets = {}
        for model, specs in self.dataset_config['Models'].items():
            for cut in specs["cuts"]: # separate dataset objects for separate cuts
                
                # Find the precentage of events in this cut
                if cut["cut_on"]!=None:
                    with uproot.open(specs['nom_input']) as dataset_file:
                        cut_var = cut["cut_on"]
                        total_df = pd.DataFrame(dataset_file["reco"][cut_var],columns=[cut_var]) 
                        cut_df = self.getCutDF(total_df,cut_var,cut["max"],cut["min"])
                        perc_events = int(100*len(cut_df)/len(total_df))
                        self.datasets.update({model+'('+cut["tag"]+')': Dataset(model,cut["color_scheme"],cut["tag"],cut_var,cut["max"],cut["min"],perc_events,specs["shortname"])})
                else:
                    self.datasets.update({model+'('+cut["tag"]+')': Dataset(model,cut["color_scheme"],reco_method_short=specs["shortname"])})
                
            
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
            
            
    def getCutDF(self,df,cut_on,max,min):
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
            print('WARNING: No cuts made.')

        return df_cut
    
    
    def getDataFrame(self,input_type,dataset_name,obs_name,extra_truth_vars=[],extra_reco_vars=[]):
        """
        Get truth and reco dataframes containing the given observable name. Also include any extra specified variables.
        
            Parameters:
                input_type (str): Type of data to put in the dataframe (i.e. 'nom', 'sysUP', or 'sysDOWN').
                dataset_name (str): Name of the dataset (e.g. "KLFitter(LL>-52)".)
                obs_name (str): Name of the observable needed (e.g. "th_pt").
            
            Optional:
                extra_truth_vars (list of str): List of any additional truth variables that will be needed.
                extra_reco_vars (list of str): List of any additional reco variables that will be needed.
            
            Return:
                df (pd.DataFrame): Dataframe containing truth and reco values for the given observable. Also includes any extra truth and reco variables that were specified.
        """
        
        # Get the reco model object
        dataset = self.datasets[dataset_name] 
        
        # Set the variables (making sure we don't have observable name in there multiple times)
        truth_vars = [obs_name] + extra_truth_vars if obs_name not in extra_truth_vars else extra_truth_vars
        reco_vars = [obs_name] + extra_reco_vars if obs_name not in extra_reco_vars else extra_reco_vars
        
        # Read in data to pandas dataframe
        with uproot.open(self.dataset_config["Models"][dataset.reco_method][input_type+"_input"]) as data_file:
            
            if input_type=="nom":  # nominal data
                reco_df = data_file["reco"].arrays(reco_vars,library="pd")
                reco_df = reco_df.add_prefix('reco_')
                truth_df = data_file["parton"].arrays(truth_vars,library="pd") # only truth data for nominal events
                truth_df = truth_df.add_prefix('truth_')
                df = pd.concat([truth_df,reco_df], axis=1)  
            else:   # systematics data
                df = data_file[input_type].arrays(reco_vars,library="pd")
                df = df.add_prefix('reco_')
                
                ### FIX FOR KLFITTER AND OTHERS (OR RATHER FIX THOSE FILES TO MATCH THIS) 
        
        return df
            

    
    def getDatasetList(self,par,var,datasets_to_plot,extra_vars=[],with_systematics=False):
        """
        Get list of datasets for plots that require multiple datasets.
        
            Parameters:
                par (str): Name of the particle (e.g.'th').
                var (str): Name of the variable (e.g.'eta').
                datasets_to_plot (list of str): List of datasets to include.
                
            Optional:
                extra_vars (list of str): List of extra variables that may be needed (e.g. for cuts) (e.g. ['pt']) (default: []).
                with_systematics (bool): Whether to include systematics in the datasets.
        """
        
        # Set the observable of interest
        obs_name = par+'_'+var
        
        # Iterate through each dataset (and save to list)
        datasets = []
        for dataset_name in datasets_to_plot:
            
            # Get the dataset object
            dataset = self.datasets[dataset_name] 
            
            # Add any extra variables we may want to cut on
            extra_truth_vars = [par+'_'+cut_var for cut_var in extra_vars]
            extra_reco_vars = [par+'_'+cut_var for cut_var in extra_vars]
            
            # Get the cut variable name for the dataset, if there is one
            if dataset.cut_var!=None:
                extra_reco_vars.append(dataset.cut_var) 
                    
            # Get the dataframe
            df = self.getDataFrame('nom',dataset_name,obs_name,extra_truth_vars=extra_truth_vars,extra_reco_vars=extra_reco_vars)
                    
            # Make cut if necessary
            if dataset.cut_var!=None:
                df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)

            # Link data to the dataset
            dataset.link_temp_df(df)
            
            # Also get systematics dataframes if required
            if with_systematics:
                up_df = self.getDataFrame('sysUP',dataset_name,obs_name,extra_truth_vars=extra_truth_vars,extra_reco_vars=extra_reco_vars)
                down_df = self.getDataFrame('sysDOWN',dataset_name,obs_name,extra_truth_vars=extra_truth_vars,extra_reco_vars=extra_reco_vars)
                if dataset.cut_var!=None:
                    up_df = self.getCutDF(up_df,dataset.cut_var,dataset.cut_max,dataset.cut_min)
                    down_df = self.getCutDF(down_df,dataset.cut_var,dataset.cut_max,dataset.cut_min)
                dataset.link_temp_sysUP_df(up_df)
                dataset.link_temp_sysDOWN_df(down_df)

            # Save to list
            datasets.append(dataset)
            
        return datasets
                 
               
    def makeTruthRecoPlots(self):
        """
        Makes truth reco plots.
        """
        
        # Get the plotting instructions
        reco_models_to_plot = self.truthreco_config["reco_models_to_plot"]
        observables_to_plot = self.truthreco_config["variables"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(reco_models_to_plot)

        # Iterate through the observables
        for par, vars in observables_to_plot.items():
            for var, specs in vars.items():
                
                # Get some important particle observable info
                obs_name = par+'_'+var
                particle = PARTICLES[par]
                observable = particle.get_observable(var)
                
                # Read the specs
                x_min = specs["min"]
                x_max = specs["max"]
                nbins = specs["nbins"]
                    
                # Iterate through each dataset
                for dataset_name in datasets_to_plot:
                    
                    # Get the dataset object
                    dataset = self.datasets[dataset_name] 
                    
                    # Get the cut variable name for the dataset, if there is one
                    extra_reco_vars = [dataset.cut_var] if dataset.cut_var!=None else []
                    
                    # Get the dataframe
                    df = self.getDataFrame('nom',dataset_name,obs_name,extra_reco_vars=extra_reco_vars)
                    
                    # Make cut if necessary
                    if dataset.cut_var!=None:
                        df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)

                    # Link data to the dataset
                    dataset.link_temp_df(df)
            
                    # Make the plot
                    Plots.TruthReco_Hist(dataset,particle,observable,x_min,x_max,nbins,self.main_dir+'/'+par+'/TruthReco/')

        print('TruthReco plots completed.')  
        
        
    def makeCMPlots(self):
        """
        Make confusion matrix plots.
        """
        
        # Get the plotting instructions
        reco_models_to_plot = self.cm_config["reco_models_to_plot"]
        observables_to_plot = self.cm_config["variables"]
        even_stats = self.cm_config["even_stats_binning"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(reco_models_to_plot)
            
        # Iterate through the observables
        for par, vars in observables_to_plot.items():
            for var, specs in vars.items():
                
                # Get some important particle observable info
                obs_name = par+'_'+var
                particle = PARTICLES[par]
                observable = particle.get_observable(var)
                
                # Read the specs and get ticks
                if even_stats:
                    nbins = specs["even_stats_bins"]["nbins"]
                    with h5py.File(self.dataset_config['Test_Data']['nom_input'],'r') as test_file:
                        temp_df = pd.DataFrame(np.array(test_file.get(obs_name)),columns=[obs_name])
                    ticks, tick_labels = Util.get_even_stats_ticks(temp_df[obs_name],observable,nbins)
                    stats_tag = '(stats_binning)'
                else:
                    x_min = specs["custom_bins"]["min"]
                    x_max = specs["custom_bins"]["max"]
                    step = specs["custom_bins"]["step"]
                    ticks, tick_labels = Util.get_ticks(observable,x_min,x_max,step)
                    stats_tag = ''
                
                # Iterate through each dataset
                for dataset_name in datasets_to_plot:
                    
                    # Get the dataset object
                    dataset = self.datasets[dataset_name] 
                    
                    # Get the cut variable name for the dataset, if there is one
                    extra_reco_vars = [dataset.cut_var] if dataset.cut_var!=None else []
                    
                    # Get the dataframe
                    df = self.getDataFrame('nom',dataset_name,obs_name,extra_reco_vars=extra_reco_vars)
                    
                    # Make cut if necessary
                    if dataset.cut_var!=None:
                        df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)

                    # Link data to the dataset
                    dataset.link_temp_df(df)
                    
                    # Make the plot
                    Plots.Confusion_Matrix(dataset,particle,observable,ticks,tick_labels,tag=stats_tag,save_loc=self.main_dir+'/'+par+'/CM/')
                    
        print('CM plots completed.')  
        
        
    def makeResPlots(self):
        """
        Make resolution/residuals plots.
        """
        
        # Get the plotting instructions
        reco_models_to_plot = self.res_config["reco_models_to_plot"]
        observables_to_plot = self.res_config["variables"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(reco_models_to_plot)
        
        # Iterate through the observables
        for par, vars in observables_to_plot.items():
            for var, specs in vars.items():
                
                # Get some important particle observable info
                particle = PARTICLES[par]
                observable = particle.get_observable(var)
                
                # Get datasets (adding pt as an extra variable if we're going to cut on it but don't already have it)
                if (len(specs["pt_cuts"])==1 and specs["pt_cuts"][0]=={}) or var=='pt':
                    datasets = self.getDatasetList(par,var,datasets_to_plot)
                else:
                    datasets = self.getDatasetList(par,var,datasets_to_plot,extra_vars=['pt'])
                
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
                    Plots.Res_Hist(cut_datasets,particle,observable,save_loc=self.main_dir+'/'+par+'/Res/',tag=pt_tag,nbins=specs["nbins"],include_moments=specs["include_moments"])
                
        print('Res plots completed.')  
        
        
        
    def makeResVsVarPlots(self):
        """
        Make residual/resolution vs. variable plots.
        """
        
        # Get the plotting instructions
        reco_models_to_plot = self.res_vs_var_config["reco_models_to_plot"]
        observables_to_plot = self.res_vs_var_config["variables"]
        even_stats = self.res_vs_var_config["even_stats_binning"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(reco_models_to_plot)
                
        # Iterate through the observables
        for par, plot_requests in observables_to_plot.items():
            for plot_specs in plot_requests:
                
                # Get some important particle observable info
                particle = PARTICLES[par]
                x_var = plot_specs["x_var"]
                y_var = plot_specs["y_var"]
                core_fit = plot_specs["core_fit"]
                x_observable = particle.get_observable(x_var)
                y_observable = particle.get_observable(y_var)
                x_obs_name = par+'_'+x_var
                
                # Get datasets
                extra_vars = [y_var] if y_var!=x_var else []
                datasets = self.getDatasetList(par,x_var,datasets_to_plot,extra_vars=extra_vars)
                
                # Read the specs and get ticks
                if even_stats:
                    nbins = plot_specs["n_even_stats_bins"]
                    with h5py.File(self.dataset_config['Test_Data']['nom_input'],'r') as test_file:
                        temp_df = pd.DataFrame(np.array(test_file.get(x_obs_name)),columns=[x_obs_name])
                    ticks, tick_labels = Util.get_even_stats_ticks(temp_df[x_obs_name],x_observable,nbins)
                    stats_tag = '(stats_binning)'
                else:
                    x_min = plot_specs["custom_bins"][0]
                    x_max = plot_specs["custom_bins"][1]
                    step = plot_specs["custom_bins"][2]
                    ticks, tick_labels = Util.get_ticks(x_observable,x_min,x_max,step)
                    stats_tag = ''

                # Make plot!
                Plots.Res_vs_Var(datasets,particle,y_observable,x_observable,ticks,tick_labels,save_loc=self.main_dir+par+'/ResVsVar/',tag=stats_tag,core_fit=core_fit)
                
        print('ResVsVar plots completed.') 
        
        
        
    def makeSysPlots(self):
        """
        Makes systematics plots.
        """
        
        # Get the plotting instructions
        reco_models_to_plot = self.sys_config["reco_models_to_plot"] # BUT WE ALSO NEED CUT VERSIONS!!!
        observables_to_plot = self.sys_config["variables"]
        even_stats = self.sys_config["even_stats_binning"]
        
        # Get list of the datasets we want to plot
        datasets_to_plot = self.getDatasetsToPlot(reco_models_to_plot)
        
        # Iterate through the observables
        for par, vars in observables_to_plot.items():
            for var, specs in vars.items():
                
                # Get some important particle observable info
                obs_name = par+'_'+var
                particle = PARTICLES[par]
                observable = particle.get_observable(var)
                
                # Get datasets
                datasets = self.getDatasetList(par,var,datasets_to_plot,with_systematics=True)
                
                # Read the specs and get ticks
                if even_stats:
                    nbins = specs["even_stats_bins"]["nbins"]
                    with h5py.File(self.dataset_config['Test_Data']['nom_input'],'r') as test_file:
                        temp_df = pd.DataFrame(np.array(test_file.get(obs_name)),columns=[obs_name])
                    ticks, tick_labels = Util.get_even_stats_ticks(temp_df[obs_name],observable,nbins)
                    stats_tag = '(stats_binning)'
                else:
                    x_min = specs["custom_bins"]["min"]
                    x_max = specs["custom_bins"]["max"]
                    step = specs["custom_bins"]["step"]
                    ticks, tick_labels = Util.get_ticks(observable,x_min,x_max,step)
                    stats_tag = ''
                
                # Make plot!
                Plots.Sys_Hist(datasets,particle,observable,ticks,tick_labels,save_loc=self.main_dir+par+'/Sys/',tag=stats_tag)
                    
                
    def makePlots(self):
        """
        Makes all desired plots.
        """
        
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
    
        
        