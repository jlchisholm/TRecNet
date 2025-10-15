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

def calculate_res(variable, df):
    """
    Calculates the resolution or residuals for a given variable.
    
        Parameters:
            variable (Variable object): Variable of interest.
            df (pd.DataFrame): Dataframe to append the data to.
            
        Returns:
            df (pd.DataFrame): Original dataframe with the resolution or residuals appended.
    """
    
    if variable.res=='Resolution':
        if variable.observable.name=='phi':
            df['res_'+variable.name] = wrap_phi((df['reco_'+variable.name] - df['truth_'+variable.name]))/df['truth_'+variable.name]
        else:
            df['res_'+variable.name] = (df['reco_'+variable.name] - df['truth_'+variable.name])/df['truth_'+variable.name]
    else:
        if variable.observable.name=='phi':
            df['res_'+variable.name] = wrap_phi(df['reco_'+variable.name] - df['truth_'+variable.name])
        else:
            df['res_'+variable.name] = df['reco_'+variable.name] - df['truth_'+variable.name]
            
    return df

def get_ticks(observable, start, stop, step, folded_bins):
    """
    Creates ticks and tick labels for bins.
    
        Parameters:
            start (int or float or double): Low edge of first bin.
            stop (int or float or double): High edge of last bin.
            step (int or float or double): Bin size.
            folded_bins (bool): Whether or not to use folded bin labels (e.g. have the last tick label for pt be infinity).
        
        Returns:
            ticks (list of numbers): List of low edges for bins.
            tick_labels (list of str): List of labels for each of the bin ticks.
    """
    
    ticks = np.arange(start,stop,step)
    tick_labels = [str(x) for x in ticks]
    
    if folded_bins:
        tick_labels[0] = '-'+r'$\infty$' if observable.name in ['eta','y','yboost','ystar'] else tick_labels[0]
        tick_labels[-1] = r'$\infty$' if observable.name!='phi' else tick_labels[-1]
    
    return ticks, tick_labels


def get_even_stats_ticks(df_col,observable,folded_bins,nbins=8):
    """
    Figures out bin width such that each bin has approximately the same number of events.
    
        Parameters:
            df_col (pd.DataFrame column): Data for one particular observable for one particular particle.
            observable (Observable object): Observable for which you want even binning.
            folded_bins (bool): Whether or not to use folded bin labels (e.g. have the last tick label for pt be infinity).
            
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
    two_dec_round = ['eta','phi','y','yboost','chi','deta','dphi','m']
    #tens_round = ['E','pout','pt','px','py','Ht','m']
    ticks = np.round(ticks, 2) if observable.name in two_dec_round else np.round(ticks,-1) 

    # Create labels for the ticks
    tick_labels = [str(x) if observable.name in two_dec_round else str(int(x)) for x in ticks]
    if folded_bins:
        tick_labels[0] = '-'+r'$\pi$' if observable.name=='phi' else '-'+r'$\infty$' if observable.name in ['eta','y','pout','px','py','yboost'] else '0'
        tick_labels[-1] = r'$\pi$' if observable.name=='phi' else r'$\infty$'

    return ticks, tick_labels


def save_plot_info(dir, fig_name, num_events, num_in, in_percent):
    """
    Writes some info about the plots to a file.
    
        Parameters:
            dir (str): Directory to save plot info in.
            fig_name (str): Name for figure.
            num_events (dictionary): Dictionary of total number of events in datasets.
            num_in (dictionary): Dictionary of total number of events in datasets, in the plotted range.
            in_percent (dictionary): Dictionary of percentage of events in datasets, in the plotted range.
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
    

    

