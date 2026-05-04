##########################################################################
#                                                                        #
#  JetPretrainer_v1.py                                                   #
#  Author: Jenna Chisholm                                                #
#  Updated: Jul.21/25                                                    #
#                                                                        #
#  Function to construct architecture for first version of               #
#  JetPretrainer. Compatible with any TRecNet model, so long as the      #
#  same jet input and other input shape are used.                        #                                                      # 
#                                                                        #
#  This is the current (as of July 2025) recommended architecture for    #
#  JetPretrainer                                                         #
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

from keras.layers import Flatten, Dense, concatenate, Dropout

def construct_JetClassifier_v1x0(Model, jet_input, other_input):
    
    flat_jets =  Flatten(name ='flattened_jets')(jet_input) 
    concat0 = concatenate([other_input, flat_jets], name = 'concat_jets_other')
    PreDense1 = Dense(256, activation='relu', name = 'dense256_1')(concat0)
    dropout_PreDense1 = Dropout(rate=0.1)(PreDense1)
    PreDense2 = Dense(256, activation='relu', name = 'dense256_2')(dropout_PreDense1)
    output = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(PreDense2)
            
    return output