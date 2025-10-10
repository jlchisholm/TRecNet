#!/bin/sh

#SBATCH --output /home/jchishol/slurm_jobs/run_plotter_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/

python source/plotting/run_plotter.py -c /home/jchishol/TRecNet/config/plotting/examples/example_plot_config.json

# done
# do some cleanup, move output files elsewhere if needed

echo 'Done :)'
