##########################################################################
#                                                                        #
#  TRecNet_ttbb_v3.py                                                    #
#  Author: Jenna Chisholm                                                #
#  Updated: Jul.21/25                                                    #
#                                                                        #
#  Function to construct architecture for fourth version of TRecNet for  #
#  ttbb. In contrast to v3, the final bbbar processor does not use the   #
#  leptonic or hadronic outputs.                                         # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

from keras.layers import Flatten, Dense, concatenate, Masking, TimeDistributed, Reshape, Multiply

def construct_TRecNet_ttbb_v3(Model,jet_input, other_input,jet_pretrain_model):
    
    # --- INITIAL JET PROCESSOR --- #
    
    Mask = Masking(Model.mask_value, name='masking_jets')(jet_input)
    Maskshape = Reshape((Model.jets_shape[1], Model.jets_shape[2]), name='reshape_masked_jets')(Mask)
    TDDense11 = TimeDistributed(Dense(128, activation='relu'), name='TDDense128')(Maskshape)
    TDDense12 = TimeDistributed(Dense(64, activation='relu'), name='TDDense64')(TDDense11)
    
    # --- INITIAL OTHER (LEP+MET) PROCESSOR --- #
    
    Dense21 = Dense(128, activation='relu', name='dense128')(other_input)
    Dense22 = Dense(64, activation='relu', name='dense64')(Dense21)
    flat_other = Flatten(name='flattened_other')(Dense22)
    
    # --- JET CLASSIFIER --- #
    
    if Model.use_JetPretraining:
        jet_pretrain_model.trainable = False                                      # Freezing the jet pretrain model (i.e. want to use the previously trained weights)
        j_weights = jet_pretrain_model([jet_input,other_input], training=False)   # Putting the inputs into the pretrain model
    else:
        flat_jets =  Flatten(name ='flattened_jets')(jet_input) 
        concat0 = concatenate([other_input, flat_jets], name = 'concat_jets_other')
        PreDense1 = Dense(256, activation='relu', name = 'dense256_1')(concat0)
        PreDense2 = Dense(256, activation='relu', name = 'dense256_2')(PreDense1) 
        j_weights = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(PreDense2)

    # --- WEIGHTED JET PROCESSOR --- #
    
    Shape_Dot = Reshape((-1,1), name='reshape')(j_weights)
    wjets = Multiply(name='weight_jets')([Shape_Dot, TDDense12])  # Weight the jets from initial jet processor by the weights from jet classifier
    TDDense13 = TimeDistributed(Dense(256, activation='relu'), name='TDDense256_1')(wjets)
    TDDense14= TimeDistributed(Dense(256, activation='relu'), name='TDDense256_2')(TDDense13)
    Flat_wjets = Flatten(name='flattened_weighted_jets')(TDDense14)
    
    # --- SECONDARY WEIGHTED JET PROCESSOR --- #
    
    TDDense_j21 = TimeDistributed(Dense(256, activation='relu'), name='TDDense256_j21')(wjets)
    TDDense_j22= TimeDistributed(Dense(256, activation='relu'), name='TDDense256_j22')(TDDense_j21)
    Flat_wjets2 = Flatten(name='flattened_weighted_jets2')(TDDense_j22)
    
    # --- FINAL PROCESSOR --- #
    
    # Concatenate the two sides
    concat = concatenate([flat_other, Flat_wjets], name = 'concat_everything')
    
    # Get leptonic ouput
    ldense1 = Dense(256, activation='relu', name='ldense256')(concat)
    ldense2 = Dense(128, activation='relu', name='ldense128')(ldense1)
    loutput = Dense(Model.lep_shape, name='lep_output')(ldense2)
    
    # Get hadronic (+ttbar) output
    hconcat = concatenate([loutput, concat])
    hdense1 = Dense(256, activation='relu', name='hdense256')(hconcat)
    hdense2 = Dense(128, activation='relu', name='hdense128')(hdense1)
    houtput = Dense(Model.had_shape+Model.ttbar_shape, name='had_output')(hdense2)
    
    # Get b+bbar output
    bdense1 = Dense(256, activation='relu', name='bdense256')(Flat_wjets2)
    bdense2 = Dense(128, activation='relu', name='bdense128')(bdense1)
    boutput = Dense(Model.bbbar_shape, name='bbbar_output')(bdense2)

    # Final output
    output = concatenate([houtput, loutput, boutput], name='final_output')
            
    return output