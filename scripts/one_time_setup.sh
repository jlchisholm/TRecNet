#!/bin/bash

#SBATCH --nodelist=atlasserv3

# do stuff to set up computing environment, variables, create output directories, cd into the working directory, print the input files names so we can double check later in the log file, etc
#conda init bash  # I think I only need this the first time I run this on a particular server ...
#conda activate py3k

cd /home/jchishol/TRecNet/
source /home/jchishol/.bashrc
#conda init bash
conda activate py3k


python --version
which python
echo $SHELL

echo 'Done :)'
