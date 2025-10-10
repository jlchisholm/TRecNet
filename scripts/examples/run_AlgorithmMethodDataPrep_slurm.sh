#!/bin/sh

#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_AlgorithmMethodDataPrep_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/


python source/plotting/AlgorithmMethodDataPrep.py --reco_method KLFitter6 --file_list /home/jchishol/TRecNet/file_lists/mntuples_mc16_6j_file_list_08_01_22.txt --save_dir /data/jchishol/mntuples_08_01_22/Results/

# done
# do some cleanup, move output files elsewhere if needed


echo 'Done :)'