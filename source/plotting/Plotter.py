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
from DataStructures import RecoModel
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
        
        for plot_type, file_name in plot_configs.items():
            if plot_type=='TruthReco':
                self.truthreco_config = json.load(open(file_name)) # oi! add more later
            elif plot_type=='CM':
                self.cm_config = json.load(open(file_name))
            
        # Create "datasets"    
        self.reco_models = {}
        for model, specs in self.dataset_config['Models'].items():
            for cut in specs["cuts"]: # separate dataset objects for separate cuts
                
                # Find the precentage of events in this cut
                if cut["cut_on"]!=None:
                    with uproot.open(specs['nom_input']) as dataset_file:
                        cut_var = specs["cut_on"]
                        total_df = pd.DataFrame(dataset_file["reco"][cut_var])                        
                        if max==None and min!=None:
                            cut_df = total_df[total_df[cut_var]>min]
                        elif max!=None and min==None:
                            cut_df = total_df[total_df[cut_var]<max]
                        elif max!=None and min!=None:
                            cut_df = total_df[total_df[cut_var]>min]
                            cut_df = cut_df[cut_df[cut_var]<max]
                        perc_events = int(100*len(cut_df)/len(total_df))
                else:
                    perc_events = 100
    
                self.reco_models.update({model: RecoModel(model,cut["color_scheme"],cut["tag"],perc_events,specs["shortname"])})
            
               
    def makeTruthRecoPlots(self):
        """
        Makes truth reco plots.
        """
        
        # Get the plotting instructions
        datasets_to_plot = self.truthreco_config["datasets_to_plot"]
        observables_to_plot = self.truthreco_config["variables"]

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

                    # Read in truth data
                    with uproot.open(self.dataset_config["Truth"]["test_file"]) as truth_file:
                        truth_df = pd.DataFrame(truth_file["parton"][obs_name],columns=['truth_'+obs_name])
                        
                    # Read in reco data
                    with uproot.open(self.dataset_config["Models"][dataset_name]["nom_input"]) as reco_file:
                        reco_df = pd.DataFrame(reco_file["reco"][obs_name],columns=['reco_'+obs_name])
                        
                    # Combine data into one dataframe
                    df = pd.concat([truth_df,reco_df], axis=1)
                    
                    # Get reco model object
                    reco_model = self.reco_models[dataset_name]
            
                    # Make the plot
                    Plots.TruthReco_Hist(df,reco_model,particle,observable,x_min,x_max,nbins,self.main_dir+'/'+par.name+'/TruthReco/')

        print('TruthReco plots created.')
        
        
        
        
    def makeCMPlots(self):
        """
        Make confusion matrix plots.
        """
        
        # Get the plotting instructions
        datasets_to_plot = self.cm_config["datasets_to_plot"]
        even_stats = self.cm_config["even_stats_binning"]
        observables_to_plot = self.cm_config["variables"]
            
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

                    # Read in truth data
                    with uproot.open(self.dataset_config["Truth"]["test_file"]) as truth_file:
                        truth_df = pd.DataFrame(truth_file["parton"][obs_name],columns=['truth_'+obs_name])
                        
                    # Read in reco data
                    with uproot.open(self.dataset_config["Models"][dataset_name]["nom_input"]) as reco_file:
                        reco_df = pd.DataFrame(reco_file["reco"][obs_name],columns=['reco_'+obs_name])
                        
                    # Combine data into one dataframe
                    df = pd.concat([truth_df,reco_df], axis=1)
                    
                    # Get reco model object
                    reco_model = self.reco_models[dataset_name]
                    
                    # Make the plot
                    Plots.Confusion_Matrix(df,reco_model,particle,observable,ticks,tick_labels,tag=tag,save_loc=self.main_dir+'/'+par.name+'/CM/')
        
               
                
    def makePlots(self):
        """
        Makes all desired plots.
        """
        
        if 'TruthReco' in self.plots_to_make:
            self.makeTruthRecoPlots()
        if 'CM' in self.plots_to_make:
            self.makeCMPlots()
        
        print('Making plots ...')    
    
        
        