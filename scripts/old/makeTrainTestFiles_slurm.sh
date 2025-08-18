#!/bin/sh

#SBATCH --output /home/jchishol/slurm_jobs/makeTrainTestFiles_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
cd /home/jchishol/TRecNet/
source /home/jchishol/.bashrc
conda activate py3k


python MLPrep.py makeTrainTestH5Files --file_list /home/jchishol/TRecNet/file_lists/ML_10j_file_list_08_01_22.txt --output /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j --split 0.85
python MLPrep.py makeTrainTestH5Files --file_list /home/jchishol/TRecNet/file_lists/sysUP_10j_file_list_08_01_22.txt --output /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_sysUP --split 0
python MLPrep.py makeTrainTestH5Files --file_list /home/jchishol/TRecNet/file_lists/sysDOWN_10j_file_list_08_01_22.txt --output /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_sysDOWN --split 0

# done
# do some cleanup, move output files elsewhere if needed


echo 'Done :)'
