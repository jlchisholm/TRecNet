#!/bin/sh

#SBATCH --mem=20G
#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_MaxMeanMachine_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/

python source/prep/MaxMeanMachine.py --input /data/jchishol/ttbb_trecnet_0825/h5_files/nom_ttbb_10jets_b1b2_train.h5  --save_dir /data/jchishol/ttbb_trecnet_0825/maxmean --b_mode b1b2

# done
# do some cleanup, move output files elsewhere if needed


echo 'Done :)'