from __main__ import * 
import re
import numpy as np

class Shape_timesteps:
    def __init__(self):
        #self.num_jet_features = None
        #self.num_jets = None
        #self.num_other_Xfeatures = None
        #self.num_Ytimesteps = None
        #self.num_Yfeatures = None
        self.mask_value = -2    # Mask non-existent jets with -2 (needs to match the network build later)


    def jet_existence_dict(self,jet_names,dataset):
        dic = {}
        jet_pt_names = list(filter(lambda a: bool(re.match('^j[0-9]+_pt$', a)), jet_names))
        for jetpt in jet_pt_names:  # For each jet
            v = np.array(dataset.get(jetpt))  # Get the jet's pt
            dic[jetpt.split('_')[0]+'_exists'] = (v>0)*1   # Mark jet as existing if pt > 0
        return dic

    
    def create_mask(self,jet_names,num_jets,num_jet_features,dataset):

        # Figure out which jets are actually existent vs. padded (probably a better way to do this)
        # Dictionary with phi keys as keys, gives 0 for non-existing jet and 1 for existing (e.g. {'j1_phi':[1,1,1,0,1,1,0,0,1,1,1,1,0,...],...})
        exist = self.jet_existence_dict(jet_names,dataset)

        # Goes through each of the jets, gets the array of existence keys (list of jn arrays, each with num_events entries)
        mask = [exist[list(exist.keys())[i]] for i in range(int(num_jets))] 

        # Stacks the mask value so first layer is the mask values for the first event, second is for second event, etc.
        samples_jets = np.stack(mask,axis=1)

        # Now they're stacked within stacks? Interesting ...
        samples_jets = samples_jets.reshape((samples_jets.shape[0], samples_jets.shape[1], 1))

        # Do the above so that we can repeat the number for each jet feature
        # So now the first array in our array is for the first event, the first array within that is for the first jet
        return np.repeat(samples_jets, num_jet_features, axis=2) # 5 feature mask
    
    


    def reshape_X(self, dataset, X_total_df):



        # Splits the names of all the jet variables (e.g. 'j1_pt', 'j1_phi', etc.) and other variables
        jet_names = list(filter(lambda a: bool(re.match('^j[0-9]+$', a.split('_')[0])), X_total_df.keys()))
        other_names = list(filter(lambda a: a not in jet_names, X_total_df.keys()))

        # Just make sure jet names are in the order we expect
        jet_names.sort(key = lambda x: x.split('_')[0])

        # Counts the number of observables for the first jet (number of jet features)
        num_jet_features = len(list(filter(lambda a: a.split('_')[0]=='j1',jet_names)))

        # The number of jets is how many jet names we have, divided by these features
        num_jets = len(jet_names)/num_jet_features

        # Number of other is just the length
        num_other_Xfeatures = len(other_names)



        # This splits X into jets and other
        X_jets_df = X_total_df[jet_names]
        X_other_df = X_total_df[other_names]

        # I guess this makes each jet into a timestep? Again this might depend on ordering ...
        #X_timestep_jets = np.stack(np.split(X_jets, self.num_jets, axis=1), axis=1)
        X_timestep_jets = np.stack(np.split(np.array(X_jets_df),num_jets,axis=1),axis=1)

        # If mask1=1, we just get the jet back. If mask1=0, we get -2 for that jet
        mask1 = self.create_mask(jet_names,num_jets,num_jet_features,dataset)
        X_timestep_jets = X_timestep_jets*mask1 + (1-mask1)*(self.mask_value) 

        # Will want arrays later so ...
        X_other = np.array(X_other_df)

        return X_timestep_jets, X_other
