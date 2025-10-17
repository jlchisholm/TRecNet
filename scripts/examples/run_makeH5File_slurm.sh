#!/bin/sh

#SBATCH --mem=40G
#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_makeH5File_%j.out

# Go to the main TRecNet directory and source the environment
TRecNet_loc=$HOME/TRecNet/
cd $TRecNet_loc
source TRecNet_env/bin/activate

# run the code
python source/prep/MLFilePrep.py makeH5File --input /data/jchishol/ttbb_trecnet_0825/pruned_varAdder_ntuples/ttbb_603192_mc20d_fullsim_pruned.root --save_dir /data/jchishol/ttbb_trecnet_0825/h5_files --tree_type nominal --var_conf config/prep/ttbb_var_names_config.json --jn 10 --b_mode b1b2 --include_jet_truths

# done
# do some cleanup, move output files elsewhere if needed
deactivate
echo 'Done :)'