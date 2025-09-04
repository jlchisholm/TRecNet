#!/bin/sh

#SBATCH --mem=20G 
#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_EventRemover_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/

python source/prep/EventRemover.py --input /data/jchishol/ttbb_trecnet_0825/varAdder_ntuples/ttbb_603192_mc20d_fullsim.root --save_dir /data/jchishol/ttbb_trecnet_0825/pruned_varAdder_ntuples --var_conf config/prep/ttbb_var_names_config.json --min_jets 4 --min_bjets 4 --remove_nonSemiLep --remove_nonsense

# done
# do some cleanup, move output files elsewhere if needed

echo 'Done :)'