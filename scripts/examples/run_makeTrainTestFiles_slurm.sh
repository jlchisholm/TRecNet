#!/bin/sh

#SBATCH --output /slurm_jobs/run_makeTrainTestFiles_%j.out

# Go to the main TRecNet directory and source the environment
TRecNet_loc=$HOME/TRecNet/
cd $TRecNet_loc
source TRecNet_env/bin/activate


python source/prep/MLFilePrep.py makeTrainTestH5Files --file_list /home/jchishol/TRecNet/file_lists/nom_ttbb_10jets_b1b2.txt --output /data/jchishol/ttbb_trecnet_0825/h5_files/nom_ttbb_10jets_b1b2 --split 0.85
#python source/prep/MLFilePrep.py makeTrainTestH5Files --file_list /home/jchishol/TRecNet/file_lists/sysUP_10j_file_list_08_01_22.txt --output /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_sysUP --split 0
#python source/prep/MLFilePrep.py makeTrainTestH5Files --file_list /home/jchishol/TRecNet/file_lists/sysDOWN_10j_file_list_08_01_22.txt --output /mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_sysDOWN --split 0

# done
# do some cleanup, move output files elsewhere if needed

deactivate
echo 'Done :)'
