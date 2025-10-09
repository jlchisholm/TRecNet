######################################################################
#                                                                    #
#  Plotter.py                                                        #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.2/25                                                 #
#                                                                    #
#  Makes plots.                                                      #
#                                                                    #
######################################################################


# Import useful packages
import numpy as np
import matplotlib
matplotlib.use('Agg')  # need for not displaying plots when running batch jobs
from matplotlib import pyplot as plt
from matplotlib import colors
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from scipy.stats import norm
from scipy.stats import cauchy
from sigfig import round
from Util import Util
import json
import Plots
import uproot
import pandas as pd
from ParticleObservables import PARTICLES
from DataStructures import Dataset
import h5py


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
                self.truthreco_config = json.load(open(file_name)) # oi! add more later
            elif plot_type=='CM':
                self.cm_config = json.load(open(file_name))
            elif plot_type=='Res':
                self.res_config = json.load(open(file_name))
            
        # Create "datasets"    
        self.datasets = {}
        for model, specs in self.dataset_config['Models'].items():
            for cut in specs["cuts"]: # separate dataset objects for separate cuts
                
                # Find the precentage of events in this cut
                if cut["cut_on"]!=None:
                    with uproot.open(specs['nom_input']) as dataset_file:
                        cut_var = specs["cut_on"]
                        total_df = pd.DataFrame(dataset_file["reco"][cut_var]) 
                        cut_df = self.getCutDF(total_df,cut_var,specs["max"],specs["min"])
                        perc_events = int(100*len(cut_df)/len(total_df))
                        self.datasets.update({model+'('+cut["tag"]+')': Dataset(model,cut["color_scheme"],cut["tag"],cut_var,specs["max"],specs["min"],perc_events,specs["shortname"])})
                else:
                    self.datasets.update({model+'('+cut["tag"]+')': Dataset(model,cut["color_scheme"],reco_method_short=specs["shortname"])})
                
            
            
            
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
    
    
    def getDataFrame(self,dataset_name,obs_name,extra_truth_vars=[],extra_reco_vars=[]):
        """
        Get truth and reco dataframes containing the given observable name. Also include any extra specified variables.
        
            Parameters:
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
        
        # Read in truth data for the observable of interest and any extra variables
        truth_vars = extra_truth_vars.append(obs_name)
        with uproot.open(self.dataset_config["Truth"]["test_file"]) as truth_file:
            truth_df = truth_file["parton"].arrays(truth_vars,library="pd")
            truth_df.add_prefix('truth_')
            
        # Read in reco data for the observable of interest and any extra variables
        reco_vars = extra_reco_vars.append(obs_name)
        with uproot.open(self.dataset_config["Models"][dataset.reco_method]["nom_input"]) as reco_file:
            reco_df = reco_file["reco"].arrays(reco_vars,library="pd")
            reco_df.add_prefix('reco_')
            
        # Concatenate and output dataframe
        df = pd.concat([truth_df,reco_df], axis=1)    
        
        return df
            

    
    def getDatasetList(self,par,var,datasets_to_plot,extra_vars=[]):
        """
        Get list of datasets for plots that require multiple datasets.
        
            Parameters:
                par (str): Name of the particle (e.g.'th').
                var (str): Name of the variable (e.g.'eta').
                datasets_to_plot (list of str): List of datasets to include.
                
            Optional:
                extra_vars (list of str): List of extra variables that may be needed (e.g. for cuts) (e.g. ['pt']).
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
            extra_reco_vars.append(dataset.cut_var)
                    
            # Get the dataframe
            df = self.getDataFrame(dataset_name,obs_name,extra_truth_vars=extra_truth_vars,extra_reco_vars=extra_reco_vars)
                    
            # Make cut if necessary
            if dataset.cut_var!=None:
                df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)

            # Get reco model object and link data
            reco_model = self.reco_models[dataset_name]
            reco_model.link_temp_df(df)
            
            # Save to list
            datasets.append(reco_model)
            
            return datasets
                 
               
    def makeTruthRecoPlots(self):
        """
        Makes truth reco plots.
        """
        
        # Get the plotting instructions
        reco_models_to_plot = self.truthreco_config["reco_models_to_plot"] # BUT WE ALSO NEED CUT VERSIONS!!!
        observables_to_plot = self.truthreco_config["variables"]
        
        # Get list of the datasets we want to plot
        # This is different from reco models, as datasets includes separate datasets for trimmed datasets (e.g. LL<-52 vs no cut)
        datasets_to_plot = []
        for name, dataset in self.datasets.items():
            if dataset.reco_method in reco_models_to_plot:
                datasets_to_plot.append(name)

        # Iterate through the observables
        for par, vars in observables_to_plot.items():
            for var, specs in vars:
                
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
                    df = self.getDataFrame(dataset_name,obs_name,extra_reco_vars=extra_reco_vars)
                    
                    # Make cut if necessary
                    if dataset.cut_var!=None:
                        df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)

                    # Get reco model object and link data
                    reco_model = self.reco_models[dataset_name]
                    reco_model.link_temp_df(df)
            
                    # Make the plot
                    Plots.TruthReco_Hist(reco_model,particle,observable,x_min,x_max,nbins,self.main_dir+'/'+par+'/TruthReco/')

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
        # This is different from reco models, as datasets includes separate datasets for trimmed datasets (e.g. LL<-52 vs no cut)
        datasets_to_plot = []
        for name, dataset in self.datasets.items():
            if dataset.reco_method in reco_models_to_plot:
                datasets_to_plot.append(name)
            
        # Iterate through the observables
        for par, vars in observables_to_plot.items():
            for var, specs in vars:
                
                # Get some important particle observable info
                obs_name = par+'_'+var
                particle = PARTICLES[par]
                observable = particle.get_observable(var)
                
                # Read the specs and get ticks
                if even_stats:
                    nbins = specs["even_stats_bins"]["nbins"]
                    # NEED TO FIND A WAY TO GET EVEN STATS BINNING
                    ticks, tick_labels = Util.get_even_stats_ticks(truth_df['truth_'+obs_name],particle,observable,nbins)
                    tag = '(stats_binning)_'
                else:
                    x_min = specs["custom_bins"]["min"]
                    x_max = specs["custom_bins"]["max"]
                    step = specs["custom_bins"]["step"]
                    ticks, tick_labels = observable.get_ticks(x_min,x_max,step)
                    tag = ''
                
                # Iterate through each dataset
                for dataset_name in datasets_to_plot:
                    
                    # Get the dataset object
                    dataset = self.datasets[dataset_name] 
                    
                    # Get the cut variable name for the dataset, if there is one
                    extra_reco_vars = [dataset.cut_var] if dataset.cut_var!=None else []
                    
                    # Get the dataframe
                    df = self.getDataFrame(dataset_name,obs_name,extra_reco_vars=extra_reco_vars)
                    
                    # Make cut if necessary
                    if dataset.cut_var!=None:
                        df = self.getCutDF(df,dataset.cut_var,dataset.cut_max,dataset.cut_min)

                    # Get reco model object and link data
                    reco_model = self.reco_models[dataset_name]
                    reco_model.link_temp_df(df)
                    
                    # Make the plot
                    Plots.Confusion_Matrix(df,reco_model,particle,observable,ticks,tick_labels,tag=tag,save_loc=self.main_dir+'/'+par+'/CM/')
                    
        print('CM plots completed.')  
        
        
    def makeResPlots(self):
        """
        Make resolution/residuals plots.
        """
        
        # Get the plotting instructions
        reco_models_to_plot = self.truthreco_config["reco_models_to_plot"]
        observables_to_plot = self.truthreco_config["variables"]
        
        # Get list of the datasets we want to plot
        # This is different from reco models, as datasets includes separate datasets for trimmed datasets (e.g. LL<-52 vs no cut)
        datasets_to_plot = []
        for name, dataset in self.datasets.items():
            if dataset.reco_method in reco_models_to_plot:
                datasets_to_plot.append(name)
        
        # Iterate through the observables
        for par, vars in observables_to_plot.items():
            for var, specs in vars:
                
                # Get some important particle observable info
                particle = PARTICLES[par]
                observable = particle.get_observable(var)
                
                # Get datasets (adding pt as an extra variable if we're going to cut on it)
                if len(specs["pt_cuts"])==1 and specs["pt_cuts"][0]=={}:
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
                            if pt_cut["pt_low"]!=None:
                                cut_df = cut_df[cut_df['truth_'+par+'_pt']>pt_cut["pt_low"]]
                            if pt_cut["pt_high"]!=None:
                                cut_df = cut_df[cut_df['truth_'+par+'_pt']<pt_cut["pt_high"]]
                                
                        # Save new cut dataset
                        cut_dataset = dataset
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
                    Plots.Res_Hist(cut_datasets,particle,observable,save_loc=self.main_dir+'/'+par+'/Res/',tag=pt_tag,nbins=specs["nbins"],core_fit=specs["core_fit"],include_moments=specs["include_moments"])
                
        print('Res plots completed.')  
        
        
               
                
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
        
        
        print('Making plots ...')    
    
        
        