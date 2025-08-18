#!/bin/sh

#SBATCH --output /home/jchishol/slurm_jobs/makePlots_%j.out
#SBATCH --nodelist=atlasserv4

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
cd /home/jchishol/TRecNet/
source /home/jchishol/.bashrc
conda activate py3k

# Make plots
python make_plots_new.py --plotting_info plot_config-main_observables_only.json 


# done
# do some cleanup, move output files elsewhere if needed


echo 'Done :)'
