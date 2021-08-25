import numpy as np 
import matplotlib.pyplot as plt
from __main__ import crop0
from transform import Transform


class Observable(object):
    def __init__(self, name, truth, value, bins):
        self.truth = truth #Is this observable a truth distribution
        self.name =name
        if type(bins) == type(1):
            _, bins = np.histogram(value, bins)
        self.bins=bins
        self.value=value #Value of this observable
        
    def populate(self, observables):
        observables.append(self)

        

def dot(x,y):
    return np.sum(x*y,axis=1).reshape((-1,1))

def cross(x,y): 
    return np.cross(x,y)

def norm(x):
    return np.sqrt(dot(x,x)).reshape((-1,1))


def fill_observables(Y, truth, Ynames):
    # TODO: Figure out binning
    observables = []
    var = {Ynames[i] : Y[:,i].reshape((-1,1)) for i in range(len(Ynames))}
    
    Observable('top_dphi', truth, np.abs(var['th_phi']-var['tl_phi']),40).populate(observables)
    
    th_px, th_py, th_pz = Transform.polar_to_cart(var['th_pt'],var['th_eta'],var['th_phi'])
    tl_px, tl_py, tl_pz = Transform.polar_to_cart(var['tl_pt'],var['tl_eta'],var['tl_phi'])
    
    th_P = np.concatenate([th_px, th_py, th_pz], axis=1)
    tl_P = np.concatenate([tl_px, tl_py, tl_pz], axis=1)
    th_p = np.sqrt(th_px**2 + th_py**2 + th_pz**2) 
    tl_p = np.sqrt(tl_px**2 + tl_py**2 + tl_pz**2)
    
    
    Observable('th_px',truth,th_px,40).populate(observables)
    Observable('th_py',truth,th_py,40).populate(observables)
    Observable('th_pz',truth,th_pz,40).populate(observables)
    Observable('tl_px',truth,tl_px,40).populate(observables)
    Observable('tl_py',truth,tl_py,40).populate(observables)
    Observable('tl_pz',truth,tl_pz,40).populate(observables)
    
    th_m, tl_m = var['th_m'], var['tl_m']
    th_E, tl_E = np.sqrt(th_m**2+th_p**2), np.sqrt(tl_m**2 + th_p**2)
    top_E = th_E + tl_E
    top_P = th_P + tl_P
    
    Observable('top_m0',truth,top_E**2 - norm(top_P)**2,40).populate(observables)
    
    th_eta, tl_eta = var['th_eta'], var['tl_eta']
    eta_cm = 0.5*(th_eta-tl_eta)
    Observable('eta_cm',truth,eta_cm,40).populate(observables) # also the negative of this
    Observable('eta_boost',truth,0.5*(th_eta+tl_eta),40).populate(observables)
    Observable('Xi_ttbar',truth,np.exp(2*np.abs(eta_cm)),40).populate(observables)
    
    ez = np.repeat(np.array([[0,0,1]]), th_eta.shape[0],axis=0)
    Observable('th_Pout',truth, dot(th_P, cross(tl_P,ez)/norm(cross(tl_P,ez))),40).populate(observables)
    Observable('tl_Pout',truth, dot(tl_P, cross(th_P,ez)/norm(cross(tl_P,ez))),40).populate(observables)
    
    Observable('pt_tot',truth, var['th_pt']+var['tl_pt'],40).populate(observables)
    
    keys = [a.name for a in observables]
    
    return dict(zip(keys, observables))

def plot_hist(true, pred):
    right = max(np.max(true.value), np.max(pred.value))
    left = min(np.min(true.value), np.min(pred.value))
    mean = np.mean(true.value-pred.value)
    std = np.std(true.value-pred.value)
    bins = np.linspace(left, right, 50)
    plt.title(true.name)
    plt.xlabel("Mean true-pred: {0}, Std true-pred: {1}".format(mean, std))
    plt.ylabel('Freqency')
    plt.hist(true.value, bins, histtype='step', color='b', label='True', density=True)
    plt.hist(pred.value, bins, histtype='step', color='r', label='Predicted', density=True)
    plt.legend()

    

    
