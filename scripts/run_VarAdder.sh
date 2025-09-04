#!/bin/sh

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
#cd /home/jchishol/TRecNet/

python source/prep/VarAdder.py --input /data/jchishol/ttbb_trecnet_0825/unprocessed_ntuples/ttbb_603192_mc20e_fullsim_0.root --save_dir /data/jchishol/ttbb_trecnet_0825/varAdder_ntuples --var_conf config/prep/ttbb_var_names_config.json --var_adder_conf config/prep/ttbb_b1b2_var_adder_config.json

# done
# do some cleanup, move output files elsewhere if needed

#cd /home/jchishol/
echo 'Done :)'