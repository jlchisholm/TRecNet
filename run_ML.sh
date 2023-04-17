#!/bin/bash

# First run container: singularity run --nv --bind /mnt/ /fast_scratch/containers/tensorflow_2.7.0-gpu.sif
# Then run enviornment: source /home/jchishol/myVenv/bin/activate
# WITHIN THAT run this script

# For training:
python machine_learning.py training --model_name TRecNet+ttbar+JetPretrain --data /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_train.h5 --xmaxmean /mnt/xrootdg/jchishol/mntuples_08_01_22/X_maxmean_variables_ttbar_ljets_10j_train.npy --ymaxmean /mnt/xrootdg/jchishol/mntuples_08_01_22/Y_maxmean_variables_ttbar_ljets_10j_train.npy --epochs 256 --patience 4 --jet_pretrain_model /home/jchishol/TRecNet/JetPretrainer/JetPretrainer_6jets_20230412_144558/JetPretrainer_6jets_20230412_144558.keras


# For testing:
#python machine_learning.py testing --model_name TRecNet+ttbar --data /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_test.h5 --xmaxmean /mnt/xrootdg/jchishol/mntuples_08_01_22/X_maxmean_variables_ttbar_ljets_10j_train.npy --ymaxmean /mnt/xrootdg/jchishol/mntuples_08_01_22/Y_maxmean_variables_ttbar_ljets_10j_train.npy --model_id TRecNet+ttbar_6jets_20230411_234414 --data_type nominal --save_loc /mnt/xrootdg/jchishol/mntuples_08_01_22/Results/