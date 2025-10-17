##########################################################################
#                                                                        #
#  run_plotter.py                                                        #
#  Author: Jenna Chisholm                                                #
#  Updated: Oct.16/25                                                    #
#                                                                        #
#  Runs plotting software and creates results plots based on config      #
#  settings.                                                             # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

# Import useful packages
import os, sys
import logging
import json
from argparse import ArgumentParser
from Particles_and_Observables import PARTICLES
from Plots import PLOT_TYPES
from Plotter import Plotter


def createPlotDirectories(dir_name, plots_to_make):
    """
    Creates directories to store the plots in, if they do not already exist.

        Parameters: 
            dir_name (str): Name you'd like to use for the plots directory.
            plots_to_make (list of str): List of plot types that will be made.

        Returns:
            'plots/'+main_dir (str): Directory to store plots and such in. 
    """
    
    # Plots directory
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
    
    # Directory name from config file (appended number so things are never overwritten)
    i = 0
    while (os.path.exists('plots/'+dir_name+'_'+str(i))):
        i+=1
    main_dir = dir_name+'_'+str(i)
    os.mkdir('plots/'+main_dir)
        
    # Configs directory
    if not os.path.exists('plots/'+main_dir+'/configs/'):
        os.mkdir('plots/'+main_dir+'/configs/')

    # Sub-directories
    for par_dir in [particle_name+'/' for particle_name in PARTICLES.keys()]: # directory for each particle
        if not os.path.exists('plots/'+main_dir+'/'+par_dir):
            os.mkdir('plots/'+main_dir+'/'+par_dir) 
        for plot_dir in [plot_type+'/' for plot_type in plots_to_make]: # directory for each plot type
            if not os.path.exists('plots/'+main_dir+'/'+par_dir+plot_dir) and plot_dir!='TrainValLoss/':
                os.mkdir('plots/'+main_dir+'/'+par_dir+plot_dir)
    
    # Do train val loss plots separately
    if not os.path.exists('plots/'+main_dir+'/TrainValLoss/'):
        os.mkdir('plots/'+main_dir+'/TrainValLoss/')
                     
    print('Plot directories established.')
    
    return 'plots/'+main_dir


def saveConfigs(dir,*configs):
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
            with open('plots/'+dir+'/configs/'+name,'w') as outfile:
                json.dump(temp_config, outfile, indent=4)

        


### ----------- MAIN ----------- ###
    
if __name__ == "__main__":
    
    # Get JSON file name of plotting info from command line and load the file
    parser = ArgumentParser()
    parser.add_argument('-c','--plotting_config', help='JSON file name (including path) that contains the plotting specifications you wish to use.', type=str, required=True)
    parser.add_argument('-l','--log_level', help='Level of logging to use.', type=str, default='WARNING',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    args = parser.parse_args()
    f_plotting_config = args.plotting_config
    plotting_config = json.load(open(f_plotting_config))
    
    # Set logging level
    logging.basicConfig(level=args.log_level.upper(), format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Extract config file names for the plots we want to make
    config_dir = plotting_config['config_files_location']
    plot_configs = {}
    for plot_type in PLOT_TYPES:
        if plotting_config[plot_type]['make_plots']:
            plot_configs.update({plot_type:config_dir+plotting_config[plot_type]['config']})
    
    # Exit if no plots to make
    if plot_configs=={}:
        logging.error('No plots to make!')
        sys.exit()
            
    # Creates save directories for plots if they don't already exist
    dir_name = plotting_config['save_loc']
    main_dir = createPlotDirectories(dir_name, plot_configs.keys())
            
    # Save all the config files
    saveConfigs(main_dir,config_dir+plotting_config['dataset_config'],*plot_configs.values())
    
    # Make the plots
    plotter = Plotter(main_dir,config_dir+plotting_config['dataset_config'],plot_configs)
    plotter.makePlots()
    
    print('Plotting complete :)')