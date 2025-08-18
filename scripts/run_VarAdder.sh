#!/bin/sh

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/

python src/prep/VarAdder.py --input /data/ttbb/trecnet/ttbb_603192_mc20e_fullsim.root --save_dir /data/jchishol/ttbb_trecnet_0825/ --var_conf config/prepping/ttbb_var_names_config.json --add_isSemiLep --add_njets --add_jet_isbtag --add_jet_isTruth --add_hadlep --add_b1b2

# done
# do some cleanup, move output files elsewhere if needed

cd /home/jchishol/
echo 'Done :)'