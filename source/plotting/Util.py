######################################################################
#                                                                    #
#  Util.py                                                           #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.2/25                                                 #
#                                                                    #
#  Utilities specifically used in the TRecNet plotting software.     #
#                                                                    #
######################################################################


# Import useful packages
import pandas as pd
import numpy as np
from pprint import pprint


def wrap_phi(var):
    var = var%(2*np.pi)
    var = var - 2*np.pi*(var > np.pi)
    return var

def calculate_res(particle, observable, df):
    
    name = particle.name+'_'+observable.name
    res = observable.res
    
    if res=='Resolution':
        if observable.name=='phi':
            df['res_'+name] = wrap_phi((df['reco_'+name] - df['truth_'+name]))/df['truth_'+name]
        else:
            df['res_'+name] = (df['reco_'+name] - df['truth_'+name])/df['truth_'+name]
    else:
        if observable.name=='phi':
            df['res_'+name] = wrap_phi(df['reco_'+name] - df['truth_'+name])
        else:
            df['res_'+name] = df['reco_'+name] - df['truth_'+name]
            
    return df

def get_ticks(observable, start, stop, step):
    
    ticks = np.arange(start,stop,step)
    tick_labels = [str(x) for x in ticks]
    tick_labels[0] = '-'+r'$\infty$' if observable.name in ['eta','y','yboost','ystar'] else tick_labels[0]
    tick_labels[-1] = r'$\infty$' if observable.name!='phi' else tick_labels[-1]
    
    return ticks, tick_labels


def get_even_stats_ticks(df_col,observable,nbins=8):
    """
    Figures out bin width such that each bin has approximately the same number of events.
    
        Parameters:
            df_col (pd.DataFrame column): Data for one particular observable for one particular particle.
            observable (Observable object): Observable for which you want even binning.
            
        Options:
            nbins (int): Number of desired bins for the histogram (default: 30).
            
        Returns:
            ticks (array): Array of the bin edges (int or float, depending on the observable).
            tick_labels (list of str): List of labels for the ticks.  
    """

    # For variables evenly distributed about zero, we'll just make sure the positive data has even split, and then use the same binning values on the negative side
    if observable.name in ['eta','phi','y','pout','px','py','yboost']:
        pos_data = df_col[df_col>=0]
        _, pos_ticks = pd.qcut(pos_data,q=int(nbins/2),retbins=True) 
        neg_ticks = -np.flip(pos_ticks[1:])
        ticks = np.concatenate((neg_ticks,pos_ticks))
    # Otherwise, just split truth events equally into nbins
    else:
        _, ticks = pd.qcut(df_col,q=nbins,retbins=True)

    # Round the ticks to nearest two decimal places (if necessary), else round to nearest tenth
    two_dec_round = ['eta','phi','y','yboost','chi','deta','dphi']
    #tens_round = ['E','pout','pt','px','py','Ht','m']
    ticks = np.round(ticks, 2) if observable.name in two_dec_round else np.round(ticks,-1) 

    # Create labels for the ticks
    tick_labels = [str(x) if observable.name in two_dec_round else str(int(x)) for x in ticks]
    tick_labels[0] = '-'+r'$\pi$' if observable.name=='phi' else '-'+r'$\infty$' if observable.name in ['eta','y','pout','px','py','yboost'] else 0
    tick_labels[-1] = r'$\pi$' if observable.name=='phi' else r'$\infty$'

    return ticks, tick_labels


def save_plot_info(dir, fig_name, num_events, num_in, in_percent):
    """
    Writes some info about the plots to a file.
    
        Parameters:
            dir
            fig_name
            num_events
            num_in
            in_percent
    """
    
    file = open(dir+'/Plot_Info.txt', "a+")
    
    file.write("\n---------------------------------------------------")
    file.write("%s: " % fig_name)
    file.write("---------------------------------------------------\n")
    file.write("Number of events in the dataset (for the cut): \n")
    pprint(num_events,stream=file)
    file.write("Number of events in the dataset in the plotted range: \n")
    pprint(num_in,stream=file)
    file.write("Percentage of the dataset's events inside the plotted range: \n")
    pprint(in_percent,stream=file)
    file.close()
    
    #df.to_string()
    

    

