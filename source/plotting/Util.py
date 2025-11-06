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
import sys
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from pprint import pprint


def wrap_phi(phi):
    """
    Wraps phi values, so they are always within +/- pi. Written by Tao Zhang.
    
        Parameter:
            phi (int or float or double): Value of phi.
            
        Returns:
            phi (float or double): Wrapped value of phi.
    """
    
    phi = phi%(2*np.pi)
    phi = phi - 2*np.pi*(phi > np.pi)
    
    return phi


def checkUnits(df,col_name,obs_units):
    """
    Checks if the values for a particular observable are of the same order as specified in the Observable definition. If not, values are converted to the specified magnitude. Only applies to energy-related (not angular) observables.
    
        Parameters:
            df (pd.DataFrame): Dataframe.
            col_name (str): Name of the desired column in the dataframe (e.g. 'truth_th_pt').
            obs_units (str): Units for the observable of interest.
            
        Returns:
            df (pd.DataFrame): Dataframe with corrected units.
    """
            
    # Only need to check units of energy-related variables (i.e. not angular variables)
    if obs_units != "":
        
        TeV_order_range = [-1,-2] # 0.1s, 0.01s
        GeV_order_range = [1,2] # 10s, 100s
        MeV_order_range = [4,5] # 10 000s, 100 000s
        keV_order_range = [7,8] # 10 000 000s, 100 000 000s
        
        # Calculate the order of each value and the modes
        order_df = np.floor(np.log10(df))
        mode_df = order_df.mode()
        
        # Modify units as asked
        if obs_units == "GeV":
            if mode_df[col_name][0] in TeV_order_range:
                df[col_name] = df[col_name]*1000
                logger.warning(col_name+' is estimated to be in units of TeV when units of '+obs_units+' were set. Scaling to the latter units. Please ensure your histogram looks correct.')
            elif mode_df[col_name][0] in GeV_order_range:
                pass
            elif mode_df[col_name][0] in MeV_order_range:
                df[col_name] = df[col_name]/1000
                logger.warning(col_name+' is estimated to be in units of MeV when units of '+obs_units+' were set. Scaling to the latter units. Please ensure your histogram looks correct.')
            elif mode_df[col_name][0] in keV_order_range:
                df[col_name] = df[col_name]/(1000000)
                logger.warning(col_name+' is estimated to be in units of keV when units of '+obs_units+' were set. Scaling to the latter units. Please ensure your histogram looks correct.')
            else:
                logger.error(col_name+' for seems to be in a strange range and units cannot be determined. Exiting program.')
                sys.exit()

        elif obs_units == "MeV":
            if mode_df[col_name][0] in TeV_order_range:
                df[col_name] = df[col_name]*1000000
                logger.warning(col_name+' is estimated to be in units of TeV when units of '+obs_units+' were set. Scaling to the latter units. Please ensure your histogram looks correct.')
            elif mode_df[col_name][0] in GeV_order_range:
                df[col_name] = df[col_name]*1000
                logger.warning(col_name+' is estimated to be in units of GeV when units of '+obs_units+' were set. Scaling to the latter units. Please ensure your histogram looks correct.')
            elif mode_df[col_name][0] in MeV_order_range:
                pass
            elif mode_df[col_name][0] in keV_order_range:
                df[col_name] = df[col_name]/(1000)
                logger.warning(col_name+' is estimated to be in units of keV when units of '+obs_units+' were set. Scaling to the latter units. Please ensure your histogram looks correct.')
            else:
                logger.error(col_name+' for seems to be in a strange range and units cannot be determined. Exiting program.')
                sys.exit()
    
    return df


def calculateRes(variable, df):
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

def getTicks(observable, start, stop, step, folded_bins):
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


def getEvenStatsTicks(df_col,observable,folded_bins,nbins=8):
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