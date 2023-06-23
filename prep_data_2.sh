#!/bin/bash

# Specify the primary folder path containing the subfolders
primary_folder="/stuff/ttbb/raw/v01_2"
save_folder="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2"

# Define a counter
count=0

# Iterate over subfolders with 'ttbb' in their names
for subfolder in "$primary_folder"/*ttbb*; do
    # Verify if the subfolder exists and is a directory
    if [[ -d "$subfolder" ]]; then
        # Check if subfolder name contains 'lplus' or 'lminus'
        if [[ "$subfolder" == *"lplus"* || "$subfolder" == *"lminus"* ]]; then
            echo "Processing subfolder: $subfolder"

            # Iterate over .root files in the subfolder
            for file in "$subfolder"/*.root; do
                # Verify if the file exists and is readable
                if [[ -f "$file" && -r "$file" ]]; then
                    # Perform actions on each file
                    echo "Processing file: $file"

                    # Prep .root files
                    python ttbbPrep_b1b2.py prepTruth --root_file "$file" --save_dir "$save_folder" --tree_name nominal_Loose

                    # Increment counter
                    count=$(( count + 1 ))

                    # Done processing file
                    echo "Done processing: $file"
                    echo "Count: $count"
                else
                    echo "File does not exist or cannot be accessed: $file"
                    continue
                fi
            done
        else
            echo "Subfolder does not have 'lplus' or 'lminus' in its name: $subfolder"
            continue
        fi
    else
        echo "Subfolder does not exist or is not a directory: $subfolder"
        continue
    fi
done
