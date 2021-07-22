from __main__ import * 
import importlib
from utilities import Utilities 
import utilities 
importlib.reload(utilities)
import re

class Shape_timesteps:
    def __init__(self):
        self.num_jet_features = None
        self.num_jets = None
        self.num_other_Xfeatures = None
        self.num_Ytimesteps = None
        self.num_Yfeatures = None
        self.mask_value = -2
    
    def create_mask(self):
        exist = Utilities.jet_existence_dict()
        mask = [exist[list(exist.keys())[i]] for i in range(int(self.num_jets))] 
        samples_jets = np.stack(mask,axis=1)
        samples_jets = samples_jets.reshape((samples_jets.shape[0], samples_jets.shape[1], 1))
        return np.repeat(samples_jets, self.num_jet_features, axis=2) # 5 feature mask
    
    
    def reshape_X(self, X_total, X_names, timestep_other=False, mask=True):
        jet_names = list(filter(lambda a: bool(re.match('^j[0-9]+$', a.split('_')[0])), X_names))
        other_names = list(filter(lambda a: a not in jet_names, X_names))
        self.num_jet_features = len(list(filter(lambda a: a.split('_')[0]=='j1',jet_names)))
        self.num_jets = len(jet_names)/self.num_jet_features
        self.num_other_Xfeatures = len(other_names)
        # mask = self.create_mask()
        X_jets = X_total[:, 0:len(jet_names)]
        X_other = X_total[:, len(jet_names):]
        X_timestep_jets = np.stack(np.split(X_jets, self.num_jets, axis=1), axis=1)
        if mask:
            mask1 = self.create_mask()
            X_timestep_jets = X_timestep_jets*mask1 + (1-mask1)*(self.mask_value) 
        if timestep_other:
            X_other = X_other.reshape(X_other.shape[0], X_other.shape[1], 1)
        return X_timestep_jets, X_other
    
    def reshape_Y(self, Y_total, Y_names):
        self.num_Yfeatures = len(list(filter(lambda a: a.split('_')[0] == Y_names[0].split('_')[0], Y_names)))
        self.num_Ytimesteps = len(Y_names)/self.num_Yfeatures
        Y_timestepped = Y_total.reshape(Y_total.shape[0], int(self.num_Ytimesteps), int(self.num_Yfeatures))
        return Y_timestepped 
    
    def reshape_X_single(self, X_total, X_names):
        totalX_jets, totalX_other = self.reshape_X(X_total, X_names, True,True)
        totalX_other = totalX_other.reshape(totalX_other.shape[0], 1, -1)
        return np.concatenate([totalX_other, totalX_jets], axis=1)
