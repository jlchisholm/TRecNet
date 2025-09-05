#!/bin/sh

#SBATCH --output /home/jchishol/slurm_jobs/run_combineH5Files_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/


python source/prep/MLFilePrep.py combineH5Files --file_list /home/jchishol/TRecNet/file_lists/nom_ttbb_10jets_b1b2.txt --output /data/jchishol/ttbb_trecnet_0825/h5_files/nom_ttbb_10jets_b1b2

# done
# do some cleanup, move output files elsewhere if needed


echo 'Done :)'
