Prep order:

0. Check that your variables names config file has the correct names for the ntuple samples your working with.
1. Run EventRemover on each file, especially to remove bad events that will lead to poor training
2. Run VarAdder on each file, to add any extra variables you may need
3. Run JetMatcher on each file, to add jet truth to your events
4. Run MLFilePrep/makeH5File on each file, to make ML-ready files
5. Run MLFilePrep/makeTrainTestH5Files on all the files, to create final train and test data files for ML (if you're not training, you can just use MLFilePrep/combineH5Files)
6. Run MaxMeanMachine on the H5 training file (if you're not training, this is not needed)