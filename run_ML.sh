#!/bin/bash

# First run container: singularity run --nv --bind /mnt/ /fast_scratch/containers/tensorflow_2.7.0-gpu.sif
# Then run enviornment: source /home/jchishol/myVenv/bin/activate
# WITHIN THAT run this script

python machine_learning.py training --model_name TRecNet --data /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_6j_train.h5 --xmaxmean /home/jchishol/TRecNet/X_maxmean_variables_ttbar_ljets_6j_train.npy --ymaxmean /home/jchishol/TRecNet/Y_maxmean_variables_ttbar_ljets_6j_train.npy --epochs 256 --patience 4
