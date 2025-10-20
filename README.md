# TRecNet

## Introduction

TRecNet is a deep neural network designed to infer the four-vectors of top and anti-top quarks from detector-level decay products in the semi-leptonic channel of top anti-top pair production (TRecNet can currently handle: $t\bar{t}$ or $t\bar{t}b\bar{b}$). It was the main work of a master's thesis, which can be found [here](https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0437237).

## Training/Testing a Model

### Data Preparation

Before testing or training a TRecNet model, the data must be formatted properly to be fed into the model. Specifically, the TRecNet framework requires the h5 file format, very specific observable names, and a dictionary of max/mean values for each of the observables. However, everything you should need to prepare your data is in the `source/prep/` directory; all that is required is your ntuples! For training a model, follow these data preparation steps:

**1. Setup environment:** Set up and activate your python enviornment, if you have not done so already. One can run the following commands:
```console
$ setup.sh
$ source TRecNet_env/bin/activate
```

**2. Create config file:** Create a `json` config file containing the names of the observables as they appear in your ntuple. Examples in `config/prep/examples` (note: the names of the left are those used in TRecNet -- do NOT change these.)

**3. Add variables:** Add any extra variables you will need, by running VarAdder.py on each of your ntuples. To do this you will also require a json config file for VarAdder (example in `config/prep/examples`). Note that new files will have the same name (and possibly overwrite the old files, depending on save directory).
```console
$ python source/prep/VarAdder.py --input <path/ntuple.root> --save_dir <save_directory_path> --var_conf config/prep/<var_names_config.json> --var_adder_conf config/prep/<var_adder_config.json>
```
**4. Remove bad events:** Remove any bad events that will lead to poor training, by running EventRemover.py on each of the ntuples (after they've gone through VarAdder). The word 'pruned' will be added to the end of the file name.
```console
$ python source/prep/EventRemover.py --input <path/ntuple.root> --save_dir <save_directory_path> --var_conf config/prep/<var_names_config.json> --min_jets <min_jets> --min_bjets <min_bjets> --remove_nonSemiLep --remove_nonsense
```
**5. Make h5 files:** Make h5 files, by running MLFilePrep.py/makeH5File on the ntuples that you have run through both VarAdder and EventRemover.
```console
$ python source/prep/MLFilePrep.py makeH5File --input <path/ntuple.root> --save_dir <save_directory_path> --tree_type nominal --var_conf config/prep/<var_names_config.json> --jn <num_jets> --b_mode <e.g.b1b2> --include_jet_truths
```
**6. Create train/test h5 files:** Combine all your h5 files together and then split them into one training file and one testing file, using MLFilePrep/makeTrainTestH5Files. For this, you will need to create a text file that lists all of the h5 files you want to use (examples in `file_lists/` directory), and decide on what percentage of events you want to go towards training+validation. This is what you will feed into TRecNet.
```console
$ python source/prep/MLFilePrep.py makeTrainTestH5Files --file_list file_lists/<file_list.txt> --output <path/output_name> --split <percent_for_training>
```
**7. Create max/mean dictionary:** Create the dictionaries of max/mean values, by running MaxMeanMachine.py on your training h5 file. You'll want to make sure `b_mode` is set to the same thing that you used in the previous steps.
```console
$ python source/prep/MaxMeanMachine.py --input <path/training_file.h5>  --save_dir <save_directory_path> --b_mode <e.g.b1b2>

```

_**Tip:**_ If at any point you're having trouble running things and you're running out of memory, you can use `source/prep/FileSplitter.py` to split up your files into smaller chunks.

_**Tip:**_ All of the above steps (except for step 1) have example scripts in the `scripts/examples` directory.

### Training a New Model

Everything to train (and test) your new model can be found in the directory `source/ml/`. Most often you'll be creating (referred to as `create` mode) a new TRecNet model or a new classifier (that can later be inserted into a TRecNet model). However, you can also unfreeze (referred to as `unfreeze` mode) TRecNet models that were previously trained with a pretrained classifier and fine-tune your network. Finally, you can hypertune (referred to as `hypertune` mode) a TRecNet or classifier model.

**1. Set up container:** Before training your model, you will ensure you have an environment with all the necessary libraries and GPU. One can create an appropriate container by running:
```console
$ source image/build.sh <container_name.sif>
```
This only needs to be done once. Then one can run the container using the following command:
```console
$ singularity run --nv --bind <directory_with_data> <directory_container_is_in>/<container_name.sif>
```
**2. Create training config:** Create a `json` file full of your training configurations. This includes data files, number of jets, model hyperparameters, etc. Examples can be found in the `config/training/examples` directory. 

**3. Select or create an architecture:** Each different model architecture has its own build file in the directory `source/ml/Models/`. Choose which one suits your needs, or create one that suits your needs.

**4. Train your model:** Run your training model in `create` mode, using the following command (in the container):
```console
$ python source/ml/run_training.py -v <model_architecture_version> -c config/training/<training_config.json> -m create
```
Note that trained models and all their information will appear in the relevant subdirectory of the `trained_models/` directory, which is not tracked by git. If you like a model enough that you want it saved in the git repository, move it to the `models/` directory.

### Validating a Model

During training, some percentage of the training data (depending on what you put in your config file), will be allotted to validation. We can take a look at the network's predictions on this set of data to validate that our model is doing what's expected and debug if necessary. To run validation, use the following command in the container (or another environment with all the necessary packages):
```console
$ python source/ml/run_validation.py -i <model_id> -d <train_data>
```
This will make predictions using the validation portion of the training data, and save some simple plots in the trained_models/<model_id>/ directory.

### Testing a Model

To test the model, use the following command in the container (or another environment with all the necessary packages):
```console
$ python source/ml/run_prediction.py -i <model_id> -d <test_data> -s <path/save_location> --testing
```
This will make predictions using the data set you provide (which should be orthogonal to the training data you use!!!), and saves them, as well as the truth values, in a root file at the desired location. If you would also like to save the scaled variables to this root file, append `--include_scaled` to your terminal command.

## Using a Model

### Data Preparation

Before using a TRecNet model, the data must be formatted properly to be fed into the model. Specifically, the TRecNet framework requires the h5 file format and very specific observable names. However, everything you should need to prepare your data is in the `source/prep/` directory; all that is required is your ntuples! For using a model, follow these data preparation steps:

**1. Setup environment:** Set up and activate your python enviornment, if you have not done so already. One can run the following commands:
```console
$ setup.sh
$ source TRecNet_env/bin/activate
```

**2. Config file:** Create a `json` config file containing the names of the observables as they appear in your ntuple. Examples in `config/prep/examples` (note: the names of the left are those used in TRecNet -- do NOT change these.)

**3. Add variables:** Add any extra variables you will need, by running VarAdder.py on each of your ntuples. To do this you will also require a json config file for VarAdder (example in `config/prep/examples`). Note that new files will have the same name (and possibly overwrite the old files, depending on save directory).
```console
$ python source/prep/VarAdder.py --input <path/ntuple.root> --save_dir <save_directory_path> --var_conf config/prep/<var_names_config.json> --var_adder_conf config/prep/<var_adder_config.json>
```
**4. Remove bad events:** Remove undesirable events (if they haven't been removed already), by running EventRemover.py on each of the ntuples (after they've gone through VarAdder). The word 'pruned' will be added to the end of the file name.
```console
$ python source/prep/EventRemover.py --input <path/ntuple.root> --save_dir <save_directory_path> --var_conf config/prep/<var_names_config.json> --min_jets <min_jets> --min_bjets <min_bjets> --remove_nonSemiLep --remove_nonsense
```
**5. Make h5 files:** Make h5 files, by running MLFilePrep.py/makeH5File on the ntuples that you have run through both VarAdder and EventRemover.
```console
$ python source/prep/MLFilePrep.py makeH5File --input <path/ntuple.root> --save_dir <save_directory_path> --tree_type nominal --var_conf config/prep/<var_names_config.json> --jn <num_jets> --b_mode <e.g.b1b2> --include_jet_truths
```
**6. Combine h5 files:** Combine all your h5 files together into one file, using MLFilePrep/makeTrainTestH5Files. For this, you will need to create a text file that lists all of the h5 files you want to use (examples in `file_lists/` directory). This is what you will feed into TRecNet.
```console
$ python source/prep/MLFilePrep.py combineH5Files --file_list file_lists/<file_list.txt> --output <path/output_name>
```

### Making Predictions

To make predictions using a trained model on a real data set (or other dataset that has no truth attached to it), we use a command very similar to that for testing the model:
```console
$ python source/ml/run_prediction.py -i <model_id> -d <data_set> -s <path/save_location> 
```
This will make predictions using the data set you provide, and saves them in a root file at the desired location. If you would also like to save the scaled variables to this root file, append `--include_scaled` to your terminal command.

## Plotting Results

Everything needed to plot the test results of your new model can be found in the directory `source/plotting/`. To make plots, follow these steps:

**1. Setup environment:** Set up and activate your python enviornment, if you have not done so already. One can run the following commands:
```console
$ setup.sh
$ source TRecNet_env/bin/activate
```

**1.5. Prepare data from algorithm-based reconstruction methods:** If you want to compare to old algorithm-based reconstruction methods, you'll need to reformat the data a bit using the following command:
```console
$ python source/plotting/AlgorithmMethodDataPrep.py --reco_method <reco_method> --file_list file_lists/<file_list.txt>  --save_dir <path/save_location> --test_file_name <path/test_file.root>
```
This will create a root file in the given save directory that contains all the data from the files in the given file list for the given reconstruction method. If a test file name is given (and this is optional), then the resulting file will only contain events that were also in the test file. Note that currently `KLFitter4`, `KLFitter6`, `PseudoTop` and `Chi2` are supported options for `reco_method`. 

**2. Create config files:** Create `json` config files for the plotting specifications, dataset specifications, and each of the types of plots you want to make. Examples in `config/plotting/examples`. Note that the plot config file sets which types of plots to make and which config files to use for each of them.

**3. Run plotter:** Run the plotting code, using the following command:
```console
$ python source/plotting/run_plotter.py -c config/plotting/<examples/plot_config.json> - l <log_level>
```
`<log_level>` can be set to 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL' for various levels of verbosity.



