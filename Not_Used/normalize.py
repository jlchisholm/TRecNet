import numpy as np
import re
import importlib
from __main__ import *

import utilities
from utilities import Utilities 
importlib.reload(utilities)

class Scale_variables:
    def __init__(self,phi_keys,dataset,crop0): 
        self.exist_dict = Utilities.jet_existence_dict(phi_keys,dataset,crop0)
        self.boxcox_ptlamb = 1
    
    def final_maxmean(self,array, names, maxmean0):
        dont = np.array([1 if 'phi' in name or 'isbtag' in name or 'DL1r' in name or 'isMA' in name or 'isTruth' in name else 0 for name in names])
        z = (array - maxmean0[:,1])/maxmean0[:,0]
        z = z*(1-dont) + array*dont
        return z

    def inverse_final_maxmean(self, array, maxmean0, names):
        dont = np.array([1 if 'phi' in name or 'isbtag' in name or 'DL1r' in name or 'isMA' in name or 'isTruth' in name else 0 for name in names])
        z = array*maxmean0[:,0] + maxmean0[:,1]
        z = z*(1-dont) + array*dont
        return z
    
    def scale_arrays(self, dataset, keys, maxmean0, crop0):
        names = []
        arrays = []
        i = 0
        while i < len(keys):  # weird way of looping through keys ...
            key = keys[i]
            ktype = keys[i].split('_')[1]   # This gives variable (e.g. 'pt')
            var = np.array(dataset.get(key))[0:crop0]
            if 'pt' in ktype: # do pt, eta, and phi triplets ...
                var1, var2 = np.array(dataset.get(keys[i+1]))[0:crop0], np.array(dataset.get(keys[i+2]))[0:crop0] # J: this would get eta and phi I think
                ptbox,px,py,eta = cart_pt_transform(var,var1,var2,self.boxcox_ptlamb)
                short = key.split('_')[0]
                names = names + [short + '_ptbox', short + '_px', short + '_py', short + '_eta']
                arrays = arrays + [ptbox, px, py, eta]
                i+=3
            elif 'phi' in ktype:  # this is just for met_phi I think
                exist = self.exist_dict[key]
                zsin, zcos = phi4_transform(var, exist)
                arrays = arrays + [zsin, zcos]
                names = names + [key +'-sin', key +'-cos']
                i+=1
            else:
                arrays.append(var)
                names.append(key)
                i+=1
        arrays = np.stack(arrays, axis=1)
        z = self.final_maxmean(arrays, names, maxmean0)
        return z, names



    
    def invscale_arrays(self, z, names, maxmean0):
        arrays = self.inverse_final_maxmean(z, maxmean0, names)
        total = []
        i = 0 
        while i < len(names):
            full_key = names[i]
            ktype = names[i].split('_')[1]
            if 'pt' in ktype:
                ptbox, px,py,pz = arrays[:,i], arrays[:,i+1], arrays[:,i+2], arrays[:,i+3]
                pt, eta, phi = inv_cart_pt_transform(ptbox, px,py,pz,self.boxcox_ptlamb)
                total = total + [pt, eta, phi]
                i+=4
            elif 'sin' in ktype:
                exist = self.exist_dict[full_key.split('-')[0]]
                zsin = arrays[:,i]
                zcos = arrays[:,i+1]
                total.append(invphi4_transform((zsin, zcos), exist))
                i+=2
            else:
                total.append(arrays[:,i])
                i+=1
        return np.stack(total,axis=1) 
                
    def test_inverse(self, keys, maxmean0):
        z, names = self.scale_arrays(keys, maxmean0)
        inverse = self.invscale_arrays(z, names, maxmean0)
        orig = [dataset.get(keys[i])[0:crop0] for i in range(len(keys))]
        orig = np.stack(orig, axis=1)
        diff = np.max(np.abs(orig - inverse),axis=0)
        return diff


def cart_pt_transform(pt, eta, phi, lamb):
    ptbox = pt 
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    return ptbox,px,py,eta

def inv_cart_pt_transform(ptbox,px,py,eta, lamb):
    pt = ptbox 
    phi = np.arctan2(py, px)
    return pt, eta, phi

def phi4_transform(arr, exist):
    w = arr % (2*np.pi)
    sin = 2/np.pi*np.arcsin(np.sin(w)) - 1.2*(1-exist)
    cos = 2/np.pi*np.arcsin(np.cos(w)) - 2.2*(1-exist)
    return (sin, cos)

def invphi4_transform(z, exist):
    pi = np.pi
    sin, cos = z[0] + 1.2*(1-exist), z[1] + 2.2*(1-exist)
    sin0, cos0 = np.sin(pi/2*sin),  np.sin(pi/2*cos)
    w = np.arctan2(sin0, cos0)
    x = w% (2*pi)
    x = x-2*np.pi*(x>pi)
    return x    

                

                

                

