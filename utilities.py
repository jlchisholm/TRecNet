from __main__ import * 
import re

class Utilities:
    def get_maxmean_dict(phi_keys,dataset,crop0): 
        to_get = [pt_keys, eta_keys, m_keys, DL1r_keys]
        keys = ['pt', 'eta', 'm','DL1r']
        maxmean= {} 
        
        for i in range(4):
            dset = to_get[i]
            for x in dset:
                arr = []
                arr.append(np.array(dataset.get(x))[0:crop0])
            arr = np.stack(arr,axis=1)
            maxmean[keys[i]] = (np.max(np.abs(arr)), np.mean(arr))

        maxmean['phi'] = (np.pi, 0)
        maxmean['met'] = (np.max(np.abs(dataset.get('met_met'))), np.mean(dataset.get('met_met')))
        return maxmean 
    
    def jet_existence_dict(phi_keys,dataset,crop0): # For all phi variables
        dic = {}
        for key in phi_keys:
            variable = key.split('_')[0]
            if bool(re.match('^j[0-9]+$', variable)): # If the variable is a jet
                v = np.array(dataset.get(variable + '_pt'))[0:crop0]
                dic[key] = (v>1)*1    # Interesting ... seems to mark a jet as existing if pt > 0
            else:
                dic[key] = np.ones(crop0, dtype=int)
        return dic

    
