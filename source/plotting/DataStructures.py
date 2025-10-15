######################################################################
#                                                                    #
#  Data.py                                                           #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.10/25                                                 #
#                                                                    #
#  Defines classes for observables, particles, and datasets, which   #
#  are used in the plotting software for TRecNet.                    #
#                                                                    #
######################################################################


# Import useful packages
import sys
import numpy as np
import pandas as pd


class Observable:
    """
    Observable object class (for plotting purposes).
    """

    def __init__(self, name, label,  unit='', res='Resolution', alt_names=[]):
        """
        Initializes a observable object.

            Parameters:
                name (str): Name of the observable (e.g.'pt')
                label (str): Label for the observable (note: this may be different than the name if you want to, for example, use Latex formatting).
                ticks (np.array): Numpy array of where you want the axes ticks placed.

            Options:
                unit (str): Units for the observable, for axis title purposes (e.g.'[GeV]', default:'').
                res (str): Specifies whether this observable should use 'Resolution' or 'Residuals' (default:'Resolution').
                alt_names (list of str): Alternate names for the observable (default: []).

            Attributes:
                name (str): Name of the observable (e.g.'pt')
                label (str): Label for the observable (note: this may be different than the name if you want to, for example, use Latex formatting).
                ticks (np.array): Numpy array of where you want the axes ticks placed.
                unit (str): Units for the observable, for axis title purposes (e.g.'[GeV]', default:'').
                alt_names (list of str): Alternate names for the observable (default: []).
                tick_labels (list): Labels for the ticks (note: these are specially specified, since sometimes one might want an infinity symbol or such)

        """

        self.name = name
        self.label = label
        self.unit = unit if unit=='' else '['+unit+']'
        self.res = res.capitalize()
        self.alt_names = alt_names


class Particle:
    """ 
    Particle object class (for plotting purposes).
    """
    
    def __init__(self, name, label, observables, alt_names=[]):
        """
        Initializes a particle object. 

            Parameters:
                name (str): Name of the particle (e.g.'th')
                label (str): Label for the particle (note: this may be different than the name if you want to, for example, use Latex formatting).
                observables (list of observable objects): List of observable objects that one could plot for this particle.

            Options:
                alt_names (list of str): Alternate names for the particle (default: []).

            Attributes:
                name (str): Name of the particle (e.g.'th').
                observables (list of observable objects): List of observable objects that one could plot for this particle.
                labels (dictionary of strings): Dictionary of the axis labels for each observable for this particle.
                labels_nounits (dictionary of strings): Dictionary of the axis labels for each observable for this particle WITHOUT units.
                alt_names (list of str): Alternate names for the particle (default: []).
        """

        self.name = name
        self.observables = {observable.name: observable for observable in observables}
        self.alt_names = alt_names

        var_names = [var.name for var in observables]
        labels = ['$'+var.label+'^{'+label+'}$ '+var.unit for var in observables]
        labels_nounits = ['$'+var.label+'^{'+label+'}$' for var in observables]
        self.labels = dict(zip(var_names,labels))
        self.labels_nounits = dict(zip(var_names,labels_nounits))
        
    def get_observable(self,name):
        
        return self.observables[name]
        


class Dataset:
    """ 
    Dataset object class.
    """

    def __init__(self, reco_method, color, cut_tag='No Cuts', cut_var=None, cut_min=0, cut_max=0, perc_events=100, reco_method_short=''):
        """
        Initializes a dataset object.

            Parameters:
                reco_method (str): Reconstruction method (e.g.'KLFitter').
                color (str): Color identifier for the plots.
            
            Options:
                cut_tag (str): Specify the cuts that were made on this dataset (default: 'No Cuts').
                cut_var
                cut_min
                cut_max
                perc_events (float or double or int): Percentage of the total number of events in this dataset (default: 100).
                reco_method_short (str): Shorthand name for the reconstruction method, to be used in plot legends (default: <reco_method>).
            
            Attributes:
                reco_method (str): Reconstruction method (e.g.'KLFitter').
                data_type (str): Level and muon type (e.g.'parton_ejets').
                color (str): Color identifier for the plots.
                perc_events (float or double or int): Fraction of the total number of events in this dataset.
                reco_method_short (str): Shorthand name for the reconstruction method, to be used in plot legends (default: <reco_method>).
                cuts (str): Specifies the cuts that were made on this dataset (e.g. 'LL>-52').
        """

        self.reco_method = reco_method
        self.color = color
        self.cut_tag = cut_tag
        self.cut_var = cut_var
        self.cut_min = cut_min
        self.cut_max = cut_max
        self.perc_events = perc_events
        self.reco_method_short = reco_method if reco_method_short=='' else reco_method_short
    
        if cut_tag == 'LL>-52':
            self.cut_label = r'$\ln\mathcal{L}$>-52'
        elif cut_tag == 'chi2<50':
            self.cut_label = r'$\chi^2$<50'
        else:
            self.cut_label = cut_tag
            
        if self.perc_events==100 and self.cut_tag!='No Cuts': 
            print('WARNING: Do you really have 100\% of events if you are making cuts? Exiting program.')
            sys.exit()
            
    def copy(self):
        
        return Dataset(self.reco_method, self.color, self.cut_tag, self.cut_var, self.cut_min, self.cut_max, self.perc_events, self.reco_method_short)
            
    def link_temp_df(self,df):
        
        self.df = df.copy()
        
    def link_temp_sysUP_df(self,up_df):
        
        self.sysUP_df = up_df.copy()

    def link_temp_sysDOWN_df(self,down_df):
        
        self.sysDOWN_df = down_df.copy()
        
    def link_train_history(self,train_history):
        
        self.train_history = train_history


