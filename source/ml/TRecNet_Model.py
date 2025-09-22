##########################################################################
#                                                                        #
#  TRecNet_Model.py                                                      #
#  Author: Jenna Chisholm                                                #
#  Updated: Sept.8/25                                                    #
#                                                                        #
#  Defines class for TRecNet model (information storage basically).      # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

import time
from source.ml.InfoGrabber import InfoGrabber


class TRecNet_Model:
    """
    A class for creating a machine learning model object, mainly to store relevant attributes of the model.
    """

    def __init__(self):
        """
        Initializes a machine learning model object.

            Attributes:
                model_v (str): Architecture version of the model to be trained (e.g. 'TRecNet_ttbb_v2').
                model_name (str): Name of the model, including architecture version (e.g. 'TRecNet_ttbb_v2+ttbar').
                model_id (str): Full model ID, which includes the model name, number of jets, and unique model number.
                n_jets (int): Number of jets the model is trained with (default: None).
                with_ttbar (bool): Whether or not to include ttbar variables.
                extra_b_mode (str): How extra b's are defined ('bbbar' or 'b1b2').
                add_jetpretrain (bool): Whether or not use a pretrained jet classifier.
                add_bbpretrain (bool): Whether or not use a pretrained bb classifier.
                unfreeze_mode (bool): Whether or not we're unfreezing a previously trained model.    
                mask_value (int): Mask value for padded jets (needs to match data prep).
                jets_shape 
                other_shape 
                had_shape 
                lep_shape
                ttbar_shape
                bbbar_shape
                xmm_file
                ymm_file
        """
        

        
        
        self.jets_shape = None
        self.other_shape = None
        self.had_shape = None
        self.lep_shape = None
        self.ttbar_shape = None
        self.bbbar_shape = None 
        
        
        
    def initialize(self, version, n_jets, add_ttbar, extra_b_mode, add_jetpretrain, add_bbpretrain, unfreeze_mode):
        
        # Derive the model name
        model_name = version
        if extra_b_mode!=None:
            model_name += '_'+extra_b_mode
        if add_ttbar:
            model_name += '+ttbar'
        if add_jetpretrain:
            model_name += '+JetPretrain'
        if add_bbpretrain:
            model_name += '+bbPretrain'
        if unfreeze_mode:
            model_name += 'Unfrozen'
            
        # Set attributes
        self.model_v = version
        self.model_name = model_name
        self.n_jets = n_jets
        self.model_id = time.strftime(model_name+"_"+str(n_jets)+"jets_%Y%m%d_%H%M%S") # Model unique save name (based on the date)

        self.mask_value = -2   # Define here so it's consist between model building and jet timestep building
        self.with_ttbar = add_ttbar
        self.extra_b_mode = extra_b_mode
        self.use_JetPretraining = add_jetpretrain
        self.use_bbPretraining = add_bbpretrain
        self.unfreeze = unfreeze_mode
        
        
    def load(self, model_id):
            
        grabber = InfoGrabber()
        grabber.check_model_trained(model_id)
        
        self.model_v = grabber.get_model_v(model_id)
        #self.model_name = grabber.get_model_name(model_id) # func doesn't exist yet
        self.model_id = model_id
        self.n_jets = grabber.get_njets(model_id)
        self.with_ttbar = grabber.get_ttbar_status(model_id)
        self.extra_b_mode = grabber.get_extra_b_mode(model_id)
        
        self.xmm_file, self.ymm_file = grabber.get_maxmean_files(model_id)
        
        
        
        
        
        