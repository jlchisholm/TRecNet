#!/bin/sh

#SBATCH --array=1-3,6,10,11,13-18,37,38,58,60,64,66,68,72-82
#SBATCH --output /home/jchishol/slurm_jobs/appendJetMatches_%A_%a.out

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
source /home/jchishol/myVenv/bin/activate
cd /home/jchishol/TRecNet/

python MLPrep.py appendJetMatches --input /mnt/xrootdg/jchishol/mntuples_08_01_22/mc16d_6j/mntuple_ljets_${SLURM_ARRAY_TASK_ID}.root --save_dir /mnt/xrootdg/jchishol/mntuples_08_01_22/mc16d_6j --dR_cut 0.4 --allow_double_matching

# done
# do some cleanup, move output files elsewhere if needed

cd /home/jchishol/slurm_jobs/
mkdir -p appendJetMatches_${SLURM_ARRAY_JOB_ID}
mv appendJetMatches_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ./appendJetMatches_${SLURM_ARRAY_JOB_ID}/

echo 'Done :)'