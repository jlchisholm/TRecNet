#!/bin/sh

#SBATCH --mem=20G
#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_MaxMeanMachine_%j.out

# Go to the main TRecNet directory and source the environment
TRecNet_loc=$HOME/TRecNet/
cd $TRecNet_loc
source TRecNet_env/bin/activate

python source/prep/MaxMeanMachine.py --input /data/jchishol/ttbb_trecnet_0825/h5_files/nom_ttbb_10jets_b1b2_train.h5  --save_dir /data/jchishol/ttbb_trecnet_0825/maxmean --b_mode b1b2

# done
# do some cleanup, move output files elsewhere if needed

deactivate
echo 'Done :)'