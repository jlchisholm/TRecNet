#!/bin/bash

# Specify the folder paths containing the .root files and h5 files
folder_path="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_copy"
h5_save_folder1="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_6jets_h5"
h5_list_file1="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_6jets_h5/list.txt"
h5_save_folder2="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_8jets_h5"
h5_list_file2="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_8jets_h5/list.txt"
h5_save_folder3="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_10jets_h5"
h5_list_file3="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_10jets_h5/list.txt"

# Define a counter
count=0

# Iterate over .root files in the folder
for file in "$folder_path"/*.root; do
    # Perform actions on each file
    echo "Processing file: $file"

    # Prep .root files
    python ttbbPrep_b1b2.py ttbbCut --root_file $file --save_dir $folder_path --tree_name nominal_Loose --num_b_tags 3 --max_b_tags  100 --n_jets_min 6 --no_name_change 
    echo "done ttbbCut"
    python ttbbPrep_b1b2.py convertKeys --root_file $file --save_dir $folder_path --tree_name nominal_Loose --new_tree_name nominal --no_name_change
    echo "done convertKeys"
    python MLPrep_bbMatcher_b1b2.py appendJetMatches --input $file --save_dir $folder_path --ignore_up_down --already_semi_lep --no_name_change
    echo "done appendJetMatches"
    python MLPrep_bbMatcher_b1b2.py appendBBMatches --input $file --save_dir $folder_path --no_name_change
    echo "done appendBBMatches"
    python MLPrep_bbMatcher_b1b2.py makeH5File --input $file --output $h5_save_folder1/H5file$count --tree_name nominal --jn 6 --ttbb
    python MLPrep_bbMatcher_b1b2.py makeH5File --input $file --output $h5_save_folder2/H5file$count --tree_name nominal --jn 8 --ttbb
    python MLPrep_bbMatcher_b1b2.py makeH5File --input $file --output $h5_save_folder3/H5file$count --tree_name nominal --jn 10 --ttbb

    echo "$h5_save_folder1/H5file${count}_20metcut_6jets.h5" >> $h5_list_file1
    echo "$h5_save_folder2/H5file${count}_20metcut_8jets.h5" >> $h5_list_file2
    echo "$h5_save_folder3/H5file${count}_20metcut_10jets.h5" >> $h5_list_file3

    # Increment counter
    count=$(( count + 1 ))

    # Done processing file
    echo "Done processing: $file"
    echo "Count: $count"
done