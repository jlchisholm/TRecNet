#!/bin/sh

#SBATCH --output /home/jchishol/slurm_jobs/run_combineH5Files_%j.out

# Go to the main TRecNet directory and source the environment
TRecNet_loc=$HOME/TRecNet/
cd $TRecNet_loc
source TRecNet_env/bin/activate


python source/prep/MLFilePrep.py combineH5Files --file_list /home/jchishol/TRecNet/file_lists/nom_ttbb_10jets_b1b2.txt --output /data/jchishol/ttbb_trecnet_0825/h5_files/nom_ttbb_10jets_b1b2

# done
# do some cleanup, move output files elsewhere if needed

deactivate
echo 'Done :)'
