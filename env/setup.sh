#!/bin/sh

# Create environment
python -m venv TRecNet_env

# Source environment
source TRecNet_env/bin/activate

# Install TRecNet dependencies
pip install awkward~=2.8.3
pip install h5py~=3.14.0
pip install matplotlib~=3.9.4
pip install mplhep~=0.3.48
pip install numpy~=1.26.4
pip install pandas~=2.3.0
pip install psutil~=7.0.0
pip install scikit-learn~=1.6.1
pip install scipy~=1.13.1
pip install seaborn~=0.13.2
pip install uproot~=5.6.2
pip install vector~=1.6.2
