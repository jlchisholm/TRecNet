#!/bin/sh

#SBATCH --mem=20G 
#SBATCH --output /home/jchishol/slurm_jobs/run_VarAdder_%j.out

# Go to the main TRecNet directory and source the environment
TRecNet_loc=$HOME/TRecNet/
cd $TRecNet_loc
source TRecNet_env/bin/activate

# run the code
python source/prep/FileSplitter.py --input /data/ttbb/trecnet/ttbb_603192_mc20e_fullsim.root --save_dir /data/jchishol/ttbb_trecnet_0825/ --var_conf config/prep/ttbb_var_names_config.json --max_events 100000

# done
# do some cleanup, move output files elsewhere if needed
deactivate
echo 'Done :)'