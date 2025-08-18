#!/bin/bash

# Specify the folder path containing the .root files
folder_path="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_copy2"
h5_save_folder="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_8jets_bbdR_h5"
h5_list_file="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_8jets_bbdR_h5/list.txt"

jet_num=8

# Define a counter
count=0

# Iterate over .root files in the folder
for file in "$folder_path"/*.root; do
    # Perform actions on each file
    echo "Processing file: $file"

    # Prep .root files
    #python ttbbPrep_adding_b1_b2.py prepTruth --root_file $file --save_dir $folder_path --tree_name nominal_Loose --no_name_change
    #echo "done prepTruth"
    python ttbbPrep_b1b2.py ttbbCut --root_file $file --save_dir $folder_path --tree_name nominal_Loose --num_b_tags 3 --max_b_tags  100 --n_jets_min 6 --no_name_change 
    echo "done ttbbCut"
    python ttbbPrep_b1b2.py convertKeys --root_file $file --save_dir $folder_path --tree_name nominal_Loose --new_tree_name nominal --no_name_change
    echo "done convertKeys"
    python MLPrep_bbMatcher_b1b2.py appendJetMatches --input $file --save_dir $folder_path --ignore_up_down --already_semi_lep --no_name_change
    echo "done appendJetMatches"
    python MLPrep_bbMatcher_b1b2.py appendBBMatches --input $file --save_dir $folder_path --no_name_change
    echo "done appendBBMatches"
    python MLPrep_bbMatcher_b1b2.py makeH5File --input $file --output $h5_save_folder/H5file$count --tree_name nominal --jn $jet_num --ttbb

    echo "$h5_save_folder/H5file${count}_20metcut_${jet_num}jets.h5" >> $h5_list_file

    # Increment counter
    count=$(( count + 1 ))

    # Done processing file
    echo "Done processing: $file"
    echo "Count: $count"
done