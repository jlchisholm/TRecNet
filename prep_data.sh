#!/bin/bash

# Specify the folder path containing the .root files
folder_path="/mnt/xrootdd/rquinn/ttbb/v01_1/user.rquinn.mc16_13TeV.411179.PhPy8EGME_ttbb_4FS_MS_lplus.SGTOP1.e7818_a875_r10201_p4514.lj.v01_1_nomtruth_out.root"
save_folder="/home/dciarniello/summer2023/TRecNet/ttbb_training_data/root_data_with_truth_b+bbar"

# Define a counter
count=0

# Iterate over .root files in the folder
for file in "$folder_path"/*.root; do
    # Perform actions on each file
    echo "Processing file: $file"

    # Prep .root files
    python ttbbPrep.py prepTruth --root_file $file --save_dir $save_folder --tree_name nominal_Loose

    # Increment counter
    count=$(( count + 1 ))

    # Done processing file
    echo "Done processing: $file"
    echo "Count: $count"
done