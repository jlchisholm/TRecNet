import numpy as np
import pandas as pd
from __main__ import *



class Scaler:
    def __init__(self):
        pass

    def final_maxmean(self, data_dic, maxmean, process):

        # Putting it into a dataframe for easier manipulation
        df = pd.DataFrame(data_dic)

        print('RAM memory percent used:', process.memory_percent())
        print('CPU percent used: ', process.cpu_percent())

        # Figure out which variables you do and don't want to scale
        dont = {key:1 if 'phi' in key or 'isbtag' in key or 'DL1r' in key or 'isTruth' in key else 0 for key in df.keys()}
        do = {key:1-i for key, i in dont.items()}

        # Get the maxs and means of the variables we're using
        maxs = {key:maxmean[key][0] for key in df.keys()}
        means = {key:maxmean[key][1] for key in df.keys()}

        # Scale the desired variables
        df = ((df-means)/maxs)*do + df*dont   # Temporarily uses a lot of memory (about 15%) but drops back after finishing

        print('RAM memory percent used:', process.memory_percent())
        print('CPU percent used: ', process.cpu_percent())

        # Converting back to array, for machine learning purposes
        #data_arr = np.array(df_final)
        #return data_arr

        return df
    


    def inverse_final_maxmean(self, data_arr, names, maxmean):

        # Putting it into a dataframe for easier manipulation
        # THIS ONLY WORKS IF DATA ARRAY IS IN THE SAME ORDERING AS THE NAMES (I THINK this is not a problem, but definitiely motivation to try to keep working with df or dict)
        df_scaled = pd.DataFrame(data_arr, columns=names)

        # Get the maxs and means of the variables and do a max mean scaling of the data
        maxs = {key:maxmean[key][0] for key in df_scaled.keys()}
        means = {key:maxmean[key][1] for key in df_scaled.keys()}
        df_unscaled = df_scaled*maxs + means

        # Figure out which variables you do and don't want to scale
        dont = {key:1 if 'phi' in key or 'isbtag' in key or 'DL1r' in key or 'isTruth' in key else 0 for key in df_scaled.keys()}
        do = {key:1-i for key, i in dont.items()}

        # Get a final dataset, with the desired variables scaled
        df_final = df_unscaled*do + df_scaled*dont

        # Converting back to dictionary
        data_dic = df_final.to_dict('list')

        return data_dic




    def scale_arrays(self, dataset, keys, maxmean,process):

        # Create a dictionary to hold all the data
        data_dic = {}

        for key in keys:
            # Will be encoding phi as px and py when pt is available
            if 'pt' in key:
                par = key.split('_')[0]
                pts, etas, phis = np.array(dataset.get(key)), np.array(dataset.get(par+'_eta')), np.array(dataset.get(par+'_phi'))
                pts, pxs, pys, etas = self.cart_pt_transform(pts, etas, phis)
                #data_dic.update({key:pts,par+'_px':pxs,par+'_py':pys,par+'_eta':etas}) # Twice as slow
                data_dic[key], data_dic[par+'_px'], data_dic[par+'_py'], data_dic[par+'_eta'] = pts, pxs, pys, etas
            # Encode met_phi with triangle wave encoding of sin and cos
            elif key=='met_phi':
                phis = np.array(dataset.get(key))
                sins, coss = self.phi_triangle_transform(phis)
                #data_dic.update({key+'-sin':sins,key+'-cos':coss})
                data_dic[key+'-sin'], data_dic[key+'-cos'] = sins, coss
            # All other variables not yet added have no changes and can just be included as is
            elif 'm' in key or 'btag' in key or 'isTruth' in key or key=='met_met':
                #data_dic.update({key:np.array(dataset.get(key))})
                data_dic[key] = np.array(dataset.get(key))

        print('RAM memory percent used:', process.memory_percent())
        print('CPU percent used: ', process.cpu_percent())

        # Do a maxmean scaling of the data
        #data_arr = self.final_maxmean(data_dic,maxmean)
        df = self.final_maxmean(data_dic,maxmean, process)

        # Get the names
        #names = list(data_dic.keys())

        #return data_arr, names

        return df
    

     
    def invscale_arrays(self, data_arr, names, maxmean):

        data_dic = self.inverse_final_maxmean(data_arr, names, maxmean)

        unscaled_dic = {}

        for name in names:
            if 'pt' in name:
                par = name.split('_')[0]
                pts, pxs, pys, etas = data_dic[name], data_dic[par+'_px'], data_dic[par+'_py'], data_dic[par+'_eta']
                pts, etas, phis = self.inv_cart_pt_transform(pts, pxs, pys, etas)
                unscaled_dic.update({name:pts,par+'_eta':etas,par+'_phi':phis})

            elif 'sin' in name:
                # He does a weird thing to check it exists? Not sure of the purpose
                sin, cos = data_dic[name], data_dic['met_phi-cos']
                phis = self.inv_phi_triangle_transform(sin,cos)
                unscaled_dic.update({'met_phi':phis})

            elif 'm' in name or 'btag' in name or 'isTruth' in name or name=='met_met':
                unscaled_dic.update({name:data_dic[name]})

        # Might not be quite ready to return it in this format but thats fine whatever
        return unscaled_dic


    # SEE IF VECTOR WILL DO THESE INSTEAD?

    def cart_pt_transform(self, pt, eta, phi):
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        return pt,px,py,eta
    
    def inv_cart_pt_transform(self, pt, px, py, eta):
        phi = np.arctan2(py, px)
        return pt, eta, phi
    
    def phi_triangle_transform(self, phi):
        w = phi % (2*np.pi)   # This might not be needed, unsure
        sin = 2/np.pi*np.arcsin(np.sin(w))   # He had like +2.2*(1-exist) etc too ... what is the purpose??
        cos = 2/np.pi*np.arcsin(np.cos(w))
        return sin, cos
    
    def inv_phi_triangle_transform(self, s, c):
        print(s)
        sin0, cos0 = np.sin(np.pi/2*s),  np.sin(np.pi/2*c)
        w = np.arctan2(sin0, cos0)
        phi = w % (2*np.pi) # This should make sure we're between 0 and 2pi
        phi = phi-2*np.pi*(phi>np.pi) # This makes sure we're between -pi and pi
        return phi   