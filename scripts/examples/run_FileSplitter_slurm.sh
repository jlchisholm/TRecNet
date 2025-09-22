#!/bin/sh

#SBATCH --mem=20G 
#SBATCH --output /home/jchishol/slurm_jobs/run_VarAdder_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/

# run the code
python source/prep/FileSplitter.py --input /data/ttbb/trecnet/ttbb_603192_mc20e_fullsim.root --save_dir /data/jchishol/ttbb_trecnet_0825/ --var_conf config/prep/ttbb_var_names_config.json --max_events 100000

# done
# do some cleanup, move output files elsewhere if needed

echo 'Done :)'