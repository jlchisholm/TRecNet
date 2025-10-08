##########################################################################
#                                                                        #
#  run_plotter.py                                                        #
#  Author: Jenna Chisholm                                                #
#  Updated: Oct.2/25                                                     #
#                                                                        #
#  Runs plotting software and creates results plots based on config      #
#  settings.                                                             # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

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
from source.plotting.Plot import *
import os, sys
from argparse import ArgumentParser
from ParticleObservables import PARTICLES
from Plotter import Plotter
from Plots import PLOT_TYPES



def createDirectories(main_dir, plots_to_make):
    """
    Creates directories to store the plots in, if they do not already exist.

        Parameters: 
            main_dir (str): Name (including path) of the primary directory you want to save the plots in.
            plots_to_make (list of str): List of plot types that will be made.

        Returns:
            Creates all the <main_dir>, with directories for each particle within, and directories for each of the plot types within each of the aforementioned particle directories.
    """
    
    # Main directory
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)

    # Sub-directories
    for par_dir in [particle_name+'/' for particle_name in PARTICLES.keys()]: # directory for each particle
        if not os.path.exists(main_dir+par_dir):
            os.mkdir(main_dir+par_dir) 
        for plot_dir in [plot_type+'/' for plot_type in plots_to_make]: # directory for each plot type
            if not os.path.exists(main_dir+par_dir+plot_dir):
                os.mkdir(main_dir+par_dir+plot_dir)
                     
    print('Directories established.')


def save_configs(dir,*configs):
    """
    Saves any number of JSON config files to the given directory.
    
        Parameters:
            dir (str): Directory to save to.
            configs (str): File name (including path) of JSON config file.
    """
    
    for f_config in configs:
        name = f_config.split('/')[-1]

        with open(f_config) as infile:
            temp_config = json.load(infile)
            with open(dir+'/'+name,'w') as outfile:
                json.dump(temp_config, outfile, indent=4)

        


### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Get JSON file name of plotting info from command line and load the file
    parser = ArgumentParser()
    parser.add_argument('-c','--plotting_config', help='JSON file name (including path) that contains the plotting specifications you wish to use.', type=str, required=True)
    args = parser.parse_args()
    f_plotting_config = args.plotting_config
    plotting_config = json.load(open(f_plotting_config))
    
    # Extract config file names for the plots we want to make
    plot_configs = {}
    for plot_type in PLOT_TYPES:
        if plotting_config[plot_type]['make_plots']:
            plot_configs.update({plot_type:plotting_config[plot_type]['config']})
            
    # Creates save directories for plots if they don't already exist
    main_dir = plotting_config['save_loc']
    createDirectories(main_dir, PARTICLES, plot_configs.keys())
            
    # Save all the config files
    save_configs(main_dir, plotting_config['dataset_config'],*plot_configs.values())
    
    # Make the plots
    plotter = Plotter(main_dir,plotting_config['dataset_config'],plot_configs)
    plotter.makePlots()
    
    
    
    

    
    print('Plotting complete :)')