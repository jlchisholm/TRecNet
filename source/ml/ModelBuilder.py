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
        print(f'Constructing architecture for model version: {self.Model.model_v}')

        # need to pass architechtural hyperparams to model constructor for some models,
        # so we look for a hparams attribute in the model and pass it if it exists
        hparams = getattr(self.Model, "hparams", None)
        if hparams is None:
            hparams = getattr(self.Model, "model_hparams", None)
        if hparams is None:
            hparams = {}

        if (self.Model.model_v == 'JetClassifier_v1'):
            from Models.JetClassifier_v1 import construct_JetClassifier_v1
            output = construct_JetClassifier_v1(self.Model, jet_input, other_input)
        elif (self.Model.model_v == 'JetClassifier_v1x0'):
            from Models.JetClassifier_v1x0 import construct_JetClassifier_v1x0
            output = construct_JetClassifier_v1x0(self.Model, jet_input, other_input)      
        elif (self.Model.model_v == 'bbClassifier_v1'):
            from Models.bbClassifier_v1 import construct_bbClassifier_v1
            output = construct_bbClassifier_v1(self.Model, jet_input, other_input)
        elif (self.Model.model_v == 'bbClassifier_v1x0'):
            from Models.bbClassifier_v1x0 import construct_bbClassifier_v1x0
            output = construct_bbClassifier_v1x0(self.Model, jet_input, other_input)
        elif (self.Model.model_v == 'TRecNet_tt_v1'):
            from Models.TRecNet_tt_v1 import construct_TRecNet_tt_v1
            output = construct_TRecNet_tt_v1(self.Model, jet_input, other_input, jet_pretrain_model)
        elif (self.Model.model_v == 'TRecNet_ttbb_v1'):
            from Models.TRecNet_ttbb_v1 import construct_TRecNet_ttbb_v1
            output = construct_TRecNet_ttbb_v1(self.Model, jet_input, other_input, jet_pretrain_model)
        elif (self.Model.model_v == 'TRecNet_ttbb_v2'):
            from Models.TRecNet_ttbb_v2 import construct_TRecNet_ttbb_v2
            output = construct_TRecNet_ttbb_v2(self.Model, jet_input, other_input, jet_pretrain_model)
        elif (self.Model.model_v == 'TRecNet_ttbb_v3'):
            from Models.TRecNet_ttbb_v3 import construct_TRecNet_ttbb_v3
            output = construct_TRecNet_ttbb_v3(self.Model, jet_input, other_input, jet_pretrain_model)
        elif (self.Model.model_v == 'TRecNet_ttbb_v4'):
            from Models.TRecNet_ttbb_v4 import construct_TRecNet_ttbb_v4
            output = construct_TRecNet_ttbb_v4(self.Model, jet_input, other_input, jet_pretrain_model)
        elif (self.Model.model_v == 'TRecNet_ttbb_v5'):
            from Models.TRecNet_ttbb_v5 import construct_TRecNet_ttbb_v5
            output = construct_TRecNet_ttbb_v5(self.Model, jet_input, other_input, jet_pretrain_model, bb_pretrain_model, hparams=hparams)
        # added TRecNet_ttbb_v*x0 _v*x* are tommy models
        elif (self.Model.model_v == 'TRecNet_ttbb_v4x0'):
            from Models.TRecNet_ttbb_v4x0 import construct_TRecNet_ttbb_v4x0
            output = construct_TRecNet_ttbb_v4x0(self.Model, jet_input, other_input, jet_pretrain_model)
        elif (self.Model.model_v == 'TRecNet_ttbb_v5x0'):
            from Models.TRecNet_ttbb_v5x0 import construct_TRecNet_ttbb_v5x0
            output = construct_TRecNet_ttbb_v5x0(self.Model, jet_input, other_input, jet_pretrain_model, bb_pretrain_model)
        elif (self.Model.model_v == 'TRecNet_ttbb_v5x1'):
            from Models.TRecNet_ttbb_v5x1 import construct_TRecNet_ttbb_v5x1
            output = construct_TRecNet_ttbb_v5x1(self.Model, jet_input, other_input, jet_pretrain_model, bb_pretrain_model, hparams=hparams)
        elif (self.Model.model_v == 'TRecNet_ttbb_v5x1_clf'):
            from Models.TRecNet_ttbb_v5x1_clf import construct_TRecNet_ttbb_v5x1_clf
            output = construct_TRecNet_ttbb_v5x1_clf(self.Model, jet_input, other_input, jet_pretrain_model, bb_pretrain_model)
        elif (self.Model.model_v == 'TRecNet_ttbb_v5x2'):
            from Models.TRecNet_ttbb_v5x2 import construct_TRecNet_ttbb_v5x2
            output = construct_TRecNet_ttbb_v5x2(self.Model, jet_input, other_input, jet_pretrain_model, bb_pretrain_model, hparams=hparams)
        elif (self.Model.model_v == 'TRecNet_ttbb_v5x3'):
            from Models.TRecNet_ttbb_v5x3 import construct_TRecNet_ttbb_v5x3
            output = construct_TRecNet_ttbb_v5x3(self.Model, jet_input, other_input, jet_pretrain_model, bb_pretrain_model)

        else: raise Exception("Unknown model version")

        return output
    
    def create_model(
        self,
        initial_lr,
        final_lr_div,
        lr_power,
        lr_decay_step,
        jet_pretrain_model=None,
        bb_pretrain_model=None,
        frozen_file=None,
        optim='adam', # added optimizer option to choose AdamW or Adam
    ):
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
            # quick validation in logs
            print("create_model: constructed new model; type:", type(model).__name__)

        # Learning rate and optimization settings
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=lr_decay_step,
            end_learning_rate=initial_lr / float(final_lr_div),
            power=lr_power,
        )
        # addeded option for AdamW optimizer
        if optim == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        elif optim == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule)
        # Compile with relevant loss functions
        if "Pretrainer" in getattr(self.Model, "model_name", ""):
            model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["mae", "mse"], jit_compile=False)
        else:
            model.compile(loss="mae", optimizer=optimizer, metrics=["mse"], jit_compile=False)

        return model