#!/bin/sh

#SBATCH --mem=40G
#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_plotter_%j.out

# Go to the main TRecNet directory and source the environment
cd "$(dirname "$(realpath $BASH_SOURCE)")"
cd ../../
source TRecNet_env/bin/activate

python source/plotting/run_plotter.py -c /home/jchishol/TRecNet/config/plotting/examples/example_plot_config.json -l INFO

# done
# do some cleanup, move output files elsewhere if needed
deactivate
echo 'Done :)'
