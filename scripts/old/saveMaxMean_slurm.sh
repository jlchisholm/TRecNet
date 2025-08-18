#!/bin/sh

#SBATCH --output /home/jchishol/slurm_jobs/saveMaxMean_%j.out
#SBATCH --mem=50G
#SBATCH --nodelist=atlasserv6

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
cd /home/jchishol/TRecNet/
source /home/jchishol/.bashrc
conda activate py3k


python MLPrep.py saveMaxMean --input /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_train.h5 --save_dir /mnt/xrootdg/jchishol/mntuples_08_01_22

# done
# do some cleanup, move output files elsewhere if needed


echo 'Done :)'