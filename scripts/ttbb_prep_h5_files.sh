#!/bin/bash

# Specify the folder path containing the .root files
folder_path="/home/dciarniello/summer2023/TRecNet/ttbb_training_data/data"
h5_save_folder="/home/dciarniello/summer2023/TRecNet/ttbb_training_data/h5_data_10jets_3btag_3cont_ttbb"
h5_list_file="/home/dciarniello/summer2023/TRecNet/ttbb_training_data/h5_data_10jets_3btag_3cont_ttbb/list.txt"

jet_num=10

# Define a counter
count=0

# Iterate over .root files in the folder
for file in "$folder_path"/*.root; do
    # Perform actions on each file
    echo "Processing file: $file"

    # Prep .root files
    python ttbbPrep.py ttbbCut --root_file $file --save_dir $folder_path --tree_name nominal_Loose --no_name_change
    echo "done ttbbCut"
    python ttbbPrep.py convertKeys --root_file $file --save_dir $folder_path --tree_name nominal_Loose --new_tree_name nominal --no_name_change
    echo "done convertKeys"
    python MLPrep.py appendJetMatches --input $file --save_dir $folder_path --ignore_up_down --already_semi_lep --no_name_change
    echo "done appendJetMatches"
    python MLPrep.py makeH5File --input $file --output $h5_save_folder/H5file$count --tree_name nominal --jn $jet_num --ttbb

    echo "$h5_save_folder/H5file${count}_metcut20_${jet_num}jets.h5" >> $h5_list_file

    # Increment counter
    count=$(( count + 1 ))

    # Done processing file
    echo "Done processing: $file"
    echo "Count: $count"
done