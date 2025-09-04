import os
import h5py
import shutil

# Input and output directories
input_directory = '/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_8jets_bbdR_h5'
output_directory = '/mnt/xrootdc/TRecNet_ttbb_training_data/ttbb_training_data/V01_2/ttbb+b1b2_8jets_bbdR_h5_fixed'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get a list of all .h5 files in the input directory
input_files = [file for file in os.listdir(input_directory) if file.endswith('.h5')]

# Process each input file
for input_file in input_files:
    # Construct the input and output file paths
    input_path = os.path.join(input_directory, input_file)
    output_path = os.path.join(output_directory, input_file)

    # Open the input file in read mode
    with h5py.File(input_path, 'r') as input_h5:
        # Create a new output file with the same name
        with h5py.File(output_path, 'w') as output_h5:
            # Iterate through the branches in the input file
            for branch in input_h5.keys():
                # Check if the branch ends with the specified suffix
                if branch.endswith('_isTruth_bb'):
                    # Get the data from the input branch
                    data = input_h5[branch][:]

                    # Modify the data by replacing 2 with 1
                    data[data == 2] = 1

                    # Create the modified branch in the output file
                    output_h5.create_dataset(branch, data=data)
                else:
                    # Copy non-modified branches to the output file
                    input_h5.copy(branch, output_h5)

print('Files copied and modified successfully!')