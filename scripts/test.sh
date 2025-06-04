#!/bin/bash

folder_path="/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+bbdphi+bbdR_10jets_h5"
output_file_name="list.txt"

# Check if folder path is provided
if [ -z "$folder_path" ]; then
  echo "Error: Folder path not provided."
  echo "Usage: ./script.sh <folder_path>"
  exit 1
fi

# Check if folder exists
if [ ! -d "$folder_path" ]; then
  echo "Error: Folder '$folder_path' does not exist."
  exit 1
fi

# Find .h5 files in the folder
h5_files=$(find "$folder_path" -type f -name "*.h5")

# Write file paths to a text file
output_file_path="$folder_path/$output_file_name"
echo "$h5_files" > "$output_file_path"

echo "File paths written to $output_file_path."