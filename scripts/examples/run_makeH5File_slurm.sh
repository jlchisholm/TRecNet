#!/bin/sh

#SBATCH --mem=40G
#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_makeH5File_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/

# run the code
python source/prep/MLFilePrep.py makeH5File --input /data/jchishol/ttbb_trecnet_0825/pruned_varAdder_ntuples/ttbb_603192_mc20d_fullsim_pruned.root --save_dir /data/jchishol/ttbb_trecnet_0825/h5_files --tree_type nominal --var_conf config/prep/ttbb_var_names_config.json --jn 10 --b_mode b1b2 --include_jet_truths

# done
# do some cleanup, move output files elsewhere if needed

echo 'Done :)'