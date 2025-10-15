#!/bin/sh

#SBATCH --mem=40G
#SBATCH --nodelist=atlas-node08
#SBATCH --output /home/jchishol/slurm_jobs/run_AlgorithmMethodDataPrep_%j.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/scratch_env/bin/activate
cd /home/jchishol/TRecNet/


#python source/plotting/AlgorithmMethodDataPrep.py --reco_method KLFitter6 --file_list /home/jchishol/TRecNet/file_lists/mntuples_mc16_6j_file_list_08_01_22.txt --save_dir /data/jchishol/mntuples_08_01_22/Results --test_file_name /data/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_test.h5
#python source/plotting/AlgorithmMethodDataPrep.py --reco_method PseudoTop --file_list /home/jchishol/TRecNet/file_lists/mntuples_mc16_6j_file_list_08_01_22.txt --save_dir /data/jchishol/mntuples_08_01_22/Results --test_file_name /data/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_test.h5
python source/plotting/AlgorithmMethodDataPrep.py --reco_method Chi2 --file_list /home/jchishol/TRecNet/file_lists/mntuples_mc16_6j_file_list_08_01_22.txt --save_dir /data/jchishol/mntuples_08_01_22/Results --test_file_name /data/jchishol/mntuples_08_01_22/variables_ttbar_ljets_10j_test.h5

# done
# do some cleanup, move output files elsewhere if needed


echo 'Done :)'