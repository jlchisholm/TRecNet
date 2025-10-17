######################################################################
#                                                                    #
#  Variables_and_Datasets.py                                         #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.16/25                                                #
#                                                                    #
#  Defines classes for variables and datasets, which are used in the #
#  plotting software for TRecNet.                                    #
#                                                                    #
######################################################################


# Import useful packages
import sys
import logging
logger = logging.getLogger(__name__)
from Particles_and_Observables import PARTICLES
  
class Variable:
    """
    Variable object class (for plotting purposes). While the Particle object contains Observable objects that it can be linked to, the Variable object is a specific particle-observable combo.
    """
    
    def __init__(self, particle, observable):
        """
        Initializes a variable object. This is what will be plotted.

            Parameters:
                particle (Particle object): Particle of the variable.
                observable (Observable object): Observable of the variable.
                
            Attributes:
                name (str): Name of the variable (e.g.'th_pt').
                particle (Particle object): Particle of the variable.
                observable (Observable object): Observable of the variable.
                res (str): Specifies whether this observable should use 'Resolution' or 'Residuals' (default:'Resolution').
                units (str): Units for the observable, for axis title purposes (e.g.'GeV', default:'').
                label (str): Axis label for the variable.
                label_nounits (str): Axis label for the variable WITHOUT units.
        """
        
        # Set attributes
        self.name = particle.name + '_' + observable.name
        self.particle = particle
        self.observable = observable
        self.res = observable.res
        self.units = observable.units
        self.label = '$'+observable.label+'^{'+particle.label+'}$ '+observable.units_label
        self.label_nounits = '$'+observable.label+'^{'+particle.label+'}$ '
        

class Dataset:
    """ 
    Dataset object class.
    """

    def __init__(self, reco_method, color, avail_vars_names, cut_tag='No Cuts', cut_var=None, cut_max=0, cut_min=0, perc_events=100, reco_method_short=''):
        """
        Initializes a dataset object.

            Parameters:
                reco_method (str): Reconstruction method (e.g.'KLFitter').
                color (str): Color identifier for the plots.
                avail_vars_names (dictionary of str): Dictionary of variables that are available to use in the data file for this dataset.
            
            Options:
                cut_tag (str): Specify the cuts that were made on this dataset (default: 'No Cuts').
                cut_var (str): Name of the variable that is to be cut on (default: None).
                cut_max (int or float or double): Maximum value for the cut variable.
                cut_min (int or float or double): Minimum value for the cut variable.
                perc_events (float or double or int): Percentage of the total number of events in this dataset (default: 100).
                reco_method_short (str): Shorthand name for the reconstruction method, to be used in plot legends (default: <reco_method>).
            
            Attributes:
                reco_method (str): Reconstruction method (e.g.'KLFitter').
                color (str): Color identifier for the plots.
                avail_vars_names (dictionary of str): Dictionary of variables that are available to use in the data file for this dataset.
                perc_events (float or double or int): Fraction of the total number of events in this dataset.
                reco_method_short (str): Shorthand name for the reconstruction method, to be used in plot legends (default: <reco_method>).
                avail_vars (dictionary of Variable objects): Dictionary of the variables that are available to use for this dataset. 
                cuts (str): Specifies the cuts that were made on this dataset (e.g. 'LL>-52').
        """

        # Set some attributes
        self.reco_method = reco_method
        self.color = color
        self.avail_vars_names = avail_vars_names
        self.cut_tag = cut_tag
        self.cut_var = cut_var
        self.cut_max = cut_max
        self.cut_min = cut_min
        self.perc_events = perc_events
        self.reco_method_short = reco_method if reco_method_short=='' else reco_method_short
    
        # Set cut label
        if cut_tag == 'LL>-52':
            self.cut_label = r'$\ln\mathcal{L}$>-52'
        elif cut_tag == 'chi2<50':
            self.cut_label = r'$\chi^2$<50'
        else:
            self.cut_label = cut_tag
             
        # Set percentage of events and make sure it makes sense           
        if self.perc_events==100 and self.cut_tag!='No Cuts': 
            logger.error('Do you really have 100\% of events if you are making cuts? Exiting program.')
            sys.exit()
        
        # Get the available variables for this dataset
        self.avail_vars = {}
        for par_name, obs in avail_vars_names.items():
            for ob_name in obs:
                
                # Get the general particle and the observable
                particle = PARTICLES[par_name]
                observable = particle.get_observable(ob_name)
                
                # Create variable and add it to the dictionary of available variables for this dataset
                variable = Variable(particle, observable)
                self.avail_vars[variable.name] = variable
       
       
    def copy(self):
        """
        Create a copy of this dataset.
        
            Returns:
                _ (Dataset object): A copy of this dataset.
        """
        
        return Dataset(self.reco_method, self.color, self.avail_vars_names, self.cut_tag, self.cut_var, self.cut_min, self.cut_max, self.perc_events, self.reco_method_short)
          
            
    def link_temp_df(self,df):
        """
        Temporarily link a dataframe to this dataset as 'df'.
        
            Parameters:
                df (pd.DataFrame): Dataframe you want to link to this dataset.
        """
        
        self.df = df.copy()
        
        
    def link_temp_sysUP_df(self,up_df):
        """
        Temporarily link an up systematics dataframe to this dataset as 'sysUP_df'.
        
            Parameters:
                up_df (pd.DataFrame): Dataframe you want to link to this dataset.
        """
        
        self.sysUP_df = up_df.copy()


    def link_temp_sysDOWN_df(self,down_df):
        """
        Temporarily link an down systematics dataframe to this dataset as 'sysDOWN_df'.
        
            Parameters:
                down_df (pd.DataFrame): Dataframe you want to link to this dataset.
        """
        
        self.sysDOWN_df = down_df.copy()
       
        
    def link_train_history(self,train_history):
        """
        Link training history to this dataset as 'train_history'.
        
            Parameters:
                train_history (np.array): Training history you want to link to this dataset.
        """
        
        self.train_history = train_history