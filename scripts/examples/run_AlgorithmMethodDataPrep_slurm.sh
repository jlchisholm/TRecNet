#!/bin/sh

#SBATCH --nodelist=atlas-node04
#SBATCH --output /home/jchishol/slurm_jobs/run_AlgorithmMethodDataPrep_%j.out

# Go to the main TRecNet directory and source the environment
TRecNet_loc=$HOME/TRecNet/
cd $TRecNet_loc
source TRecNet_env/bin/activate

#python source/plotting/AlgorithmMethodDataPrep.py --reco_method KLFitter6 --file_list /home/jchishol/TRecNet/file_lists/mntuples_mc16_6j_file_list_08_01_22.txt --save_dir /data/jchishol/mntuples_08_01_22/Results --test_file_name /data/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_test.h5
#python source/plotting/AlgorithmMethodDataPrep.py --reco_method PseudoTop --file_list /home/jchishol/TRecNet/file_lists/mntuples_mc16_6j_file_list_08_01_22.txt --save_dir /data/jchishol/mntuples_08_01_22/Results --test_file_name /data/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_test.h5
python source/plotting/AlgorithmMethodDataPrep.py --reco_method Chi2 --file_list /home/jchishol/TRecNet/file_lists/mntuples_mc16_6j_file_list_08_01_22.txt --save_dir /data/jchishol/mntuples_08_01_22/Results --test_file_name /data/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_test.h5

# done
# do some cleanup, move output files elsewhere if needed

deactivate
echo 'Done :)'