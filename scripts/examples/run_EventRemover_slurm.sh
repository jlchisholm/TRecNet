#!/bin/sh

#SBATCH --mem=20G 
#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_EventRemover_%j.out

# Go to the main TRecNet directory and source the environment
TRecNet_loc=$HOME/TRecNet/
cd $TRecNet_loc
source TRecNet_env/bin/activate

python source/prep/EventRemover.py --input /data/jchishol/ttbb_trecnet_0825/varAdder_ntuples/ttbb_603192_mc20a_fullsim_WP4.root --save_dir /data/jchishol/ttbb_trecnet_0825/pruned_varAdder_ntuples --var_conf config/prep/ttbb_var_names_config.json --min_jets 4 --min_bjets 2 --remove_nonSemiLep --remove_nonsense
python source/prep/EventRemover.py --input /data/jchishol/ttbb_trecnet_0825/varAdder_ntuples/ttbb_603192_mc20d_fullsim_WP4.root --save_dir /data/jchishol/ttbb_trecnet_0825/pruned_varAdder_ntuples --var_conf config/prep/ttbb_var_names_config.json --min_jets 4 --min_bjets 2 --remove_nonSemiLep --remove_nonsense
python source/prep/EventRemover.py --input /data/jchishol/ttbb_trecnet_0825/varAdder_ntuples/ttbb_603192_mc20e_fullsim_WP4.root --save_dir /data/jchishol/ttbb_trecnet_0825/pruned_varAdder_ntuples --var_conf config/prep/ttbb_var_names_config.json --min_jets 4 --min_bjets 2 --remove_nonSemiLep --remove_nonsense


# done
# do some cleanup, move output files elsewhere if needed
deactivate
echo 'Done :)'