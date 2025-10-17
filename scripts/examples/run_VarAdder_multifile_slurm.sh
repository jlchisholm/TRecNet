#!/bin/sh

#SBATCH --mem=40G
#SBATCH --nodelist=atlas-node08
#SBATCH --array=0-20
#SBATCH --output /home/jchishol/slurm_jobs/run_VarAdder_%A/run_VarAdder_%A_%a.out

# Go to the main TRecNet directory and source the environment
TRecNet_loc=$HOME/TRecNet/
cd $TRecNet_loc
source TRecNet_env/bin/activate

# run the code
python source/prep/VarAdder.py --input /data/jchishol/ttbb_trecnet_0825/unprocessed_ntuples/ttbb_603192_mc20e_fullsim_${SLURM_ARRAY_TASK_ID}.root --save_dir /data/jchishol/ttbb_trecnet_0825/varAdder_ntuples --var_conf config/prep/ttbb_var_names_config.json --var_adder_conf config/prep/ttbb_b1b2_var_adder_config.json

# done
# do some cleanup, move output files elsewhere if needed
deactivate
echo 'Done :)'