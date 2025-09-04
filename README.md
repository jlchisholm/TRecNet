# TRecNet

## Introduction

TRecNet is a deep neural network designed to infer the four-vectors of top and anti-top quarks from detector-level decay products in the semi-leptonic channel of top anti-top pair production. It was the main work of a master's thesis, which can be found [here](https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0437237).

## Training/Testing a Model

### Data Preparation

Before testing or training a TRecNet model, the data must be formatted properly to be fed into the model. Specifically, the TRecNet framework requires the h5 file format, very specific observable names, and a dictionary of max/mean values for each of the observables. However, everything you should need to prepare your data is in the `source/prep/` directory; all that is required is your ntuples! For training a model, follow these data preparation steps:

1. Create a `json` config file containing the names of the observables as they appear in your ntuple. Examples in `config/prep` (note: the names of the left are those used in TRecNet -- do NOT change these.)
2. Add any extra variables you will need, by running VarAdder.py on each of your ntuples. To do this you will also require a json config file for VarAdder (example in `config/prep`). Note that new files will have the same name (and possibly overwrite the old files, depending on save directory).
3. Remove any bad events that will lead to poor training, by running EventRemover.py on each of the ntuples (after they've gone through VarAdder). The word 'pruned' will be added to the end of the file name.
4. Make h5 files, by running MLFilePrep.py/makeH5File on the ntuples that you have run through both VarAdder and EventRemover.
5. Combine all your h5 files together and then split them into one training file and one testing file, using MLFilePrep/makeTrainTestH5Files. For this, you will need to create a text file that lists all of the h5 files you want to use (examples in `file_lists/` directory). This is what you will feed into TRecNet.
6. Create the dictionaries of max/mean values, by running MaxMeanMachine.py on your training h5 file.

**Tip:** If at any point you're having trouble running things and you're running out of memory, you can use `source/prep/FileSplitter.py` to split up your files into smaller chunks.
**Tip:** All of the above steps (except for step 1) have example scripts in the `scripts` directory.



## Using a Model
