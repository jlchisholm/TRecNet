# TRecNet

## Introduction

TRecNet is a deep neural network designed to infer the four-vectors of top and anti-top quarks from detector-level decay products in the semi-leptonic channel of top anti-top pair production (TRecNet can currently handle: $t\bar{t}$ or $t\bar{t}b\bar{b}$). It was the main work of a master's thesis, which can be found [here](https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0437237).

## Training/Testing a Model

### Data Preparation

Before testing or training a TRecNet model, the data must be formatted properly to be fed into the model. Specifically, the TRecNet framework requires the h5 file format, very specific observable names, and a dictionary of max/mean values for each of the observables. However, everything you should need to prepare your data is in the `source/prep/` directory; all that is required is your ntuples! For training a model, follow these data preparation steps:

**1. Config file:** Create a `json` config file containing the names of the observables as they appear in your ntuple. Examples in `config/prep` (note: the names of the left are those used in TRecNet -- do NOT change these.)
**2. Add variables:** Add any extra variables you will need, by running VarAdder.py on each of your ntuples. To do this you will also require a json config file for VarAdder (example in `config/prep`). Note that new files will have the same name (and possibly overwrite the old files, depending on save directory).
```console
$ python source/prep/VarAdder.py --input <path/ntuple.root> --save_dir <save_directory_path> --var_conf config/prep/<var_names_config.json> --var_adder_conf config/prep/<var_adder_config.json>
```
**3. Remove bad events:** Remove any bad events that will lead to poor training, by running EventRemover.py on each of the ntuples (after they've gone through VarAdder). The word 'pruned' will be added to the end of the file name.
```console
$ python source/prep/EventRemover.py --input <path/ntuple.root> --save_dir <save_directory_path> --var_conf config/prep/<var_names_config.json> --min_jets <min_jets> --min_bjets <min_bjets> --remove_nonSemiLep --remove_nonsense
```
**4. Make h5 files:** Make h5 files, by running MLFilePrep.py/makeH5File on the ntuples that you have run through both VarAdder and EventRemover.
```console
$ python source/prep/MLFilePrep.py makeH5File --input <path/ntuple.root> --save_dir <save_directory_path> --tree_type nominal --var_conf config/prep/<var_names_config.json> --jn <num_jets> --extra_b_mode <e.g.b1b2> --include_jet_truths
```
**5. Create train/test h5 files:** Combine all your h5 files together and then split them into one training file and one testing file, using MLFilePrep/makeTrainTestH5Files. For this, you will need to create a text file that lists all of the h5 files you want to use (examples in `file_lists/` directory). This is what you will feed into TRecNet.
```console
$ python source/prep/MLFilePrep.py combineH5Files --file_list file_lists/<file_list.txt> --output <path/output_name>
```
**6. Create max/mean dictionary:** Create the dictionaries of max/mean values, by running MaxMeanMachine.py on your training h5 file. You'll want to make sure `extra_b_mode` is set to the same thing that you used in the previous steps.
```console
$ python source/prep/MaxMeanMachine.py --input <path/training_file.h5>  --save_dir <save_directory_path> --extra_b_mode <e.g.b1b2>

```

_**Tip:**_ If at any point you're having trouble running things and you're running out of memory, you can use `source/prep/FileSplitter.py` to split up your files into smaller chunks.

_**Tip:**_ All of the above steps (except for step 1) have example scripts in the `scripts` directory.

### Training A New Model

Everything to train (and test) your new model can be found in the directory `source/ml/`. Most often you'll be creating (referred to as `create` mode) a new TRecNet model or a new classifier (that can later be inserted into a TRecNet model). However, you can also unfreeze (referred to as `unfreeze` mode) TRecNet models that were previously trained with a pretrained classifier and fine-tune your network. Finally, you can hypertune (referred to as `hypertune` mode) a TRecNet or classifier model.

**1. Set up container:** Before training your model, you will ensure you have an environment with all the necessary libraries and GPU. One can create an appropriate container by running:
```console
$ source image/build.sh <container_name.sif>
```
This only needs to be done once. Then one can run the container using the following command:
```console
$ singularity run --nv --bind <directory_with_data> <directory_container_is_in>/<container_name.sif>
```
**2. Create training config:** Create a `json` file full of your training configurations. This includes data files, number of jets, model hyperparameters, etc. Examples can be found in the `config/training/` directory. 

**3. Select or create an architecture:** Each different model architecture has its own build file in the directory `source/ml/Models/`. Choose which one suits your needs, or create one that suits your needs.

**4. Train your model:** Run your training model in `create` mode, using the following command (in the container):
```console
$ python source/ml/run_training.py -v <model_architecture_version> -c config/training/<training_config.json> -m create
```






## Using a Model
