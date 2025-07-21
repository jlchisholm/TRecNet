##########################################################################
#                                                                        #
#  JetPretrainer_v1.py                                                   #
#  Author: Jenna Chisholm                                                #
#  Updated: Jul.21/25                                                    #
#                                                                        #
#  Function to construct architecture for first version of               #
#  JetPretrainer. Compatible with any TRecNet model, so long as the      #
#  same jet input shape (both number of jets and number of attributes)   #
#  is used.                                                              # 
#                                                                        #
#  This is the current (as of July 2025) recommended architecture for    #
#  JetPretrainer                                                         #
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

from keras.layers import Flatten, Dense, concatenate

def construct_JetPretrainer_v1(Model, jet_input, other_input):
    
    flat_jets =  Flatten(name ='flattened_jets')(jet_input) 
    concat0 = concatenate([other_input, flat_jets], name = 'concat_jets_other')
    PreDense1 = Dense(256, activation='relu', name = 'dense256_1')(concat0)
    PreDense2 = Dense(256, activation='relu', name = 'dense256_2')(PreDense1) 
    output = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(PreDense2)
            
    return output