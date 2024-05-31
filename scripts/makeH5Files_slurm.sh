#!/bin/sh

#SBATCH --array=1-6,8-22,24-30
#SBATCH --output /home/jchishol/slurm_jobs/makeH5Files_%A_%a.out
#SBATCH --nodelist=atlasserv3
#SBATCH --mem=4G

# do stuff to set up computing environment
source /home/jchishol/myVenv/bin/activate
cd /home/jchishol/TRecNet/

# check this server is using the right version of python
printf 'Using Python version:'
python --version
which python


# make the H5 files
python MLPrep.py makeH5File --input /mnt/xrootdg/jchishol/mntuples_08_01_22/mc16e_6j/mntuple_ljets_${SLURM_ARRAY_TASK_ID}_jetMatch04.root --output /mnt/xrootdg/jchishol/mntuples_08_01_22/mc16e_6j/variables_ttbar_ljets_${SLURM_ARRAY_TASK_ID} --tree_name 'nominal' --jn 10 --met_cut 20
python MLPrep.py makeH5File --input /mnt/xrootdg/jchishol/mntuples_08_01_22/mc16e_6j/mntuple_ljets_${SLURM_ARRAY_TASK_ID}_jetMatch04.root --output /mnt/xrootdg/jchishol/mntuples_08_01_22/mc16e_6j/variables_ttbar_ljets_${SLURM_ARRAY_TASK_ID} --tree_name 'CategoryReduction_JET_Pileup_RhoTopology__1up' --jn 10 --met_cut 20
python MLPrep.py makeH5File --input /mnt/xrootdg/jchishol/mntuples_08_01_22/mc16e_6j/mntuple_ljets_${SLURM_ARRAY_TASK_ID}_jetMatch04.root --output /mnt/xrootdg/jchishol/mntuples_08_01_22/mc16e_6j/variables_ttbar_ljets_${SLURM_ARRAY_TASK_ID} --tree_name 'CategoryReduction_JET_Pileup_RhoTopology__1down' --jn 10 --met_cut 20


# done
# do some cleanup, move output files elsewhere if needed

cd /home/jchishol/slurm_jobs/
mkdir -p makeH5Files_${SLURM_ARRAY_JOB_ID}
mv makeH5Files_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ./makeH5Files_${SLURM_ARRAY_JOB_ID}/

echo 'Done :)'
