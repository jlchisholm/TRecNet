##########################################################################
#                                                                        #
#  TRecNet_ttbb_vx0.py                                                    #
#  Author: Tommy Lubomirski
#  Original Author: Jenna Chisholm                                        #
#  Updated: Sep.28/25                                                    #
#                                                                        #
#  Function to construct architecture for fourth version of TRecNet for  #
#  ttbb. In contrast to v3, the final bbbar processor does not use the   #
#  leptonic or hadronic outputs.                                         # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

from keras.layers import Flatten, Dense, concatenate, Masking, TimeDistributed, Reshape, Multiply, Dropout, LayerNormalization


def construct_TRecNet_ttbb_v4x0(Model,jet_input, other_input,jet_pretrain_model):
    
    # --- INITIAL JET PROCESSOR --- #
    
    Mask = Masking(Model.mask_value, name='masking_jets')(jet_input)
    Maskshape = Reshape((Model.jets_shape[1], Model.jets_shape[2]), name='reshape_masked_jets')(Mask)
    xj = TimeDistributed(LayerNormalization(), name='LN_jets_0')(Maskshape)
    TDDense11 = TimeDistributed(Dense(128, activation='gelu'), name='TDDense128')(xj)
    dropout11 = Dropout(0.1, name='DO_jets_128')(TDDense11)
    xj = TimeDistributed(LayerNormalization(), name='LN_jets_1')(dropout11)
    TDDense12 = TimeDistributed(Dense(64, activation='gelu'), name='TDDense64')(xj)

    # --- INITIAL OTHER (LEP+MET) PROCESSOR --- #
    xo = LayerNormalization(name='LN_other_0')(other_input)
    Dense21 = Dense(128, activation='gelu', name='dense128')(xo)
    dropout21 = Dropout(rate=0.1)(Dense21)
    Dense22 = Dense(64, activation='gelu', name='dense64')(dropout21)
    flat_other = Dense22 #Flatten(name='flattened_other')(Dense22)
    
    # --- JET CLASSIFIER --- #
    
    if Model.use_JetPretraining:
        jet_pretrain_model.trainable = False                                      # Freezing the jet pretrain model (i.e. want to use the previously trained weights)
        j_weights = jet_pretrain_model([jet_input,other_input], training=False)   # Putting the inputs into the pretrain model
    else:
        flat_jets =  Flatten(name ='flattened_jets')(jet_input) 
        concat0 = concatenate([other_input, flat_jets], name = 'concat_jets_other')
        xc = LayerNormalization(name='LN_jet_classifier')(concat0)
        PreDense1 = Dense(256, activation='gelu', name = 'dense256_1')(xc)
        jcdrop1 = Dropout(0.1, name='DO_jc_256')(PreDense1)
        PreDense2 = Dense(256, activation='gelu', name = 'dense256_2')(jcdrop1) 
        jcdrop2 = Dropout(0.1, name='DO_jc_256_2')(PreDense2)
        j_weights = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(PreDense2)

    # --- WEIGHTED JET PROCESSOR --- #
    
    Shape_Dot = Reshape((-1,1), name='reshape')(j_weights)
    wjets = Multiply(name='weight_jets')([Shape_Dot, TDDense12])  # Weight the jets from initial jet processor by the weights from jet classifier
    wjets = wjets + TDDense12   # residual path allows “amplify” behavior
    TDDense13 = TimeDistributed(Dense(256, activation='gelu'), name='TDDense256_1')(wjets)
    dropout13 = Dropout(rate=0.1)(TDDense13)
    TDDense14= TimeDistributed(Dense(256, activation='gelu'), name='TDDense256_2')(dropout13)
    Flat_wjets = Flatten(name='flattened_weighted_jets')(TDDense14)
    
    # --- SECONDARY WEIGHTED JET PROCESSOR --- #
    
    TDDense_j21 = TimeDistributed(Dense(256, activation='gelu'), name='TDDense256_j21')(wjets)
    dropout_j21 = Dropout(0.1, name='DO_jets2_256')(TDDense_j21)
    TDDense_j22= TimeDistributed(Dense(256, activation='gelu'), name='TDDense256_j22')(dropout_j21)
    Flat_wjets2 = Flatten(name='flattened_weighted_jets2')(TDDense_j22)
    
    # --- FINAL PROCESSOR --- #
    
    # Concatenate the two sides
    concat = concatenate([flat_other, Flat_wjets], name = 'concat_everything')
    concat = LayerNormalization(name='LN_concat')(concat)
    
    # Get leptonic ouput
    ln_lep = LayerNormalization(name='LN_ldense_in')(concat)
    ldense1 = Dense(256, activation='gelu', name='ldense256')(ln_lep)
    ldense1 = Dropout(0.1, name='DO_ldense256')(ldense1)
    ldense2 = Dense(128, activation='gelu', name='ldense128')(ldense1)
    loutput = Dense(Model.lep_shape, name='lep_output')(ldense2)
    
    # Get hadronic (+ttbar) output
    hconcat = concatenate([loutput, concat], name='concat_hlep')
    hconcat = LayerNormalization(name='LN_hconcat')(hconcat)
    hdense1 = Dense(256, activation='gelu', name='hdense256')(hconcat)
    hdense1 = Dropout(0.1, name='DO_hdense256')(hdense1)
    hdense2 = Dense(128, activation='gelu', name='hdense128')(hdense1)
    houtput = Dense(Model.had_shape+Model.ttbar_shape, name='had_output')(hdense2)

    # Get b+bbar output
    bb_in = LayerNormalization(name='LN_bb_in')(Flat_wjets2)
    bdense1 = Dense(256, activation='gelu', name='bdense256')(bb_in)
    dropout_bdense1 = Dropout(rate=0.1)(bdense1)
    bdense2 = Dense(128, activation='gelu', name='bdense128')(dropout_bdense1)
    boutput = Dense(Model.bbbar_shape, name='bbbar_output')(bdense2)

    # Final output
    output = concatenate([houtput, loutput, boutput], name='final_output')
            
    return output