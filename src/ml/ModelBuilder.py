##########################################################################
#                                                                        #
#  ModelBuilder.py                                                       #
#  Author: Jenna Chisholm                                                #
#  Updated: Jul.3/25                                                     #
#                                                                        #
#  Defines classes and functions relevant for to building TRecNet        #
#  models.                                                               # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

import keras
from keras.layers import Input, TFSMLayer
from keras import regularizers 
from keras import initializers


class TRecNet_Model:
    """
    A class for creating a machine learning model object, mainly to store relevant attributes of the model.
    """

    def __init__(self, model_name, model_id=None, n_jets=None):
        """
        Initializes a machine learning model object.

            Parameters:
                model_name (str): Name of the model (e.g. 'TRecNet+ttbar').
                model_id (str): Unique model identifier (default: None).
                n_jets (int): Number of jets the model is trained with (default: None).

            Attributes:
                model_name (str): Name of the model (e.g. 'TRecNet+ttbar').
                model_id (str): Unique model identifier, based on model name, number of jets, and time it was created.
                mask_value (int): 
                n_jets (int): Number of jets the model is trained with.
        """
        
        self.model_name = model_name
        self.model_id = time.strftime(model_name+"_"+str(n_jets)+"jets_%Y%m%d_%H%M%S") if model_id==None else model_id   # Model unique save name (based on the date)
        self.n_jets = n_jets if model_id==None else int(model_id.split('_')[1].split('jets')[0]) # If not given, get from model_id
        self.mask_value = -2   # Define here so it's consist between model building and jet timestep building
        
        self.pretrainer = True if 'Pretrainer' in model_name else False
        self.use_JetPretaining = True if '+JetPretrain' in model_name else False
        self.use_bbPretraining = True if '+bbPretrain' in model_name else False
        self.unfreeze = True if 'Unfrozen' in model_name else False
        self.for_ttbb = True if 'ttbb' in model_name else False
        
        self.jets_shape = None
        self.other_shape = None
        self.had_shape = None
        self.lep_shape = None
        self.ttbar_shape = None
        self.bbbar_shape = None 

class ModelBuilder:
    """
    A class for building TRecNet models.
    
        Parameters:
            Model (TRecNet_Model): TRecNet_Model object that stores important parameters of the model's architecture.
    """  
    
    def __init__(self, Model):
        self.Model = Model

    def construct_input_layers(self):
        
        jet_input = Input(shape=(self.Model.jets_shape[1], self.Model.jets_shape[2]),name='jet_input')
        other_input = Input(shape=(self.Model.other_shape[1],),name='other_input')
        
        return jet_input, other_input
        
    def construct_architecture(self, jet_input, other_input, jet_pretrain_model, bb_pretrain_model):
        
        if ('JetPretrainer_v1' in self.Model.model_name):
            from Models.JetPretrainer_v1 import construct_JetPretrainer_v1
            output = construct_JetPretrainer_v1(self.Model, jet_input, other_input)       
        elif ('TRecNet_tt_v1' in self.Model.model_name):
            from Models.TRecNet_tt_v1 import construct_TRecNet_tt_v1
            output = construct_TRecNet_tt_v1(self.Model, jet_input, other_input, jet_pretrain_model)
        elif ('TRecNet_ttbb_v1' in self.Model.model_name):
            from Models.TRecNet_ttbb_v1 import construct_TRecNet_ttbb_v1
            output = construct_TRecNet_ttbb_v1(self.Model, jet_input, other_input, jet_pretrain_model)
        elif ('TRecNet_ttbb_v2' in self.Model.model_name):
            from Models.TRecNet_ttbb_v2 import construct_TRecNet_ttbb_v2
            output = construct_TRecNet_ttbb_v2(self.Model, jet_input, other_input, jet_pretrain_model)
        elif ('TRecNet_ttbb_v3' in self.Model.model_name):
            from Models.TRecNet_ttbb_v3 import construct_TRecNet_ttbb_v3
            output = construct_TRecNet_ttbb_v3(self.Model, jet_input, other_input, jet_pretrain_model)
        elif ('TRecNet_ttbb_v4' in self.Model.model_name):
            from Models.TRecNet_ttbb_v4 import construct_TRecNet_ttbb_v4
            output = construct_TRecNet_ttbb_v4(self.Model, jet_input, other_input, jet_pretrain_model)
        elif ('TRecNet_ttbb_v5' in self.Model.model_name):
            from Models.TRecNet_ttbb_v5 import construct_TRecNet_ttbb_v5
            output = construct_TRecNet_ttbb_v5(self.Model, jet_input, other_input, jet_pretrain_model, bb_pretrain_model)
            
        return output
    
    def create_model(self, initial_lr, final_lr_div, lr_power, lr_decay_step, jet_pretrain_model=None, bb_pretrain_model=None, frozen_file=None):
        
        # For TRecNet+ttbar+JetPretrainUnfrozen we read in the model, but for others we construct it
        if self.Model.unfreeze:
            
            # Load the frozen model to start with
            model = TFSMLayer(frozen_file, call_endpoint="serving_default")
            
            # Find the jet pre-training layer and unfreeze all those sublayers
            for layer in model.layers:
                if isinstance(layer, keras.Model):
                    layer.trainable = True 
            
        else:
    
            # Construct the model's architecture
            jet_input, other_input = self.construct_input_layers()
            output = self.construct_architecture(jet_input, other_input, jet_pretrain_model, bb_pretrain_model)
            model = keras.models.Model(inputs=[jet_input, other_input], outputs=output)
             
             
        # Learning rate and optimization settings
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr, decay_steps=lr_decay_step,end_learning_rate=initial_lr/final_lr_div,power=lr_power)
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile with relevant loss functions
        if 'Pretrainer' in self.Model.model_name:
            model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['mae','mse'],jit_compile=False)
        else:
            model.compile(loss='mae', optimizer= optimizer, metrics=['mse'],jit_compile=False)
                

        return model 
