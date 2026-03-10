
#!/bin/sh


# Go to the main TRecNet directory
cd "$(dirname "$(realpath $BASH_SOURCE)")"

# Create the environment and activate it
python -m venv TRecNet_env
source TRecNet_env/bin/activate

# Add necessary packages
pip install awkward~=2.8.3
pip install h5py~=3.14.0
pip install ipykernel~=6.30.1
pip install matplotlib~=3.9.4
pip install numpy~=2.0.2
pip install pandas~=2.3.0
pip install scikit-learn~=1.6.1
pip install scipy~=1.13.1
pip install sigfig~=1.3.19
pip install tf2onnx~=1.16.1
pip install tk~=0.1.0
pip install uproot~=5.6.2
pip install vector~=1.6.2
pip install tensorflow~=2.20.0
pip install onnxruntime~=1.19.2

# Deactivate environment
deactivate