##########################################################################
#                                                                        #
#  TRecNet_ttbb_v5.py                                                    #
#  Author: Jenna Chisholm                                                #
#  Updated: Jul.21/25                                                    #
#                                                                        #
#  Function to construct architecture for fifth version of TRecNet for   #
#  ttbb.                                                                 # 
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

from keras.layers import Flatten, Dense, concatenate, Masking, TimeDistributed, Reshape, Multiply, LayerNormalization, Dropout

def construct_TRecNet_ttbb_v5x0(Model,jet_input, other_input,jet_pretrain_model,bb_pretrain_model):
    
    # --- INITIAL JET PROCESSOR --- #
    
    Mask = Masking(Model.mask_value, name='masking_jets')(jet_input)
    Maskshape = Reshape((Model.jets_shape[1], Model.jets_shape[2]), name='reshape_masked_jets')(Mask)
    TDDense11 = TimeDistributed(Dense(128, activation='gelu'), name='TDDense128')(Maskshape)
    TDDense12 = TimeDistributed(Dense(64, activation='gelu'), name='TDDense64')(TDDense11)

    # --- INITIAL BBBAR JET PROCESSOR --- #
    
    bMask = Masking(Model.mask_value, name='b_masking_jets')(jet_input)
    bMaskshape = Reshape((Model.jets_shape[1], Model.jets_shape[2]), name='b_reshape_masked_jets')(bMask)
    bTDDense11 = TimeDistributed(Dense(128, activation='gelu'), name='TDDense128_b1')(bMaskshape)
    bTDDense12 = TimeDistributed(Dense(128, activation='gelu'), name='TDDense128_b2')(bTDDense11)
    bTDDense13 = TimeDistributed(Dense(64, activation='gelu'), name='TDDense128_b3')(bTDDense12)
    
    # --- INITIAL OTHER (LEP+MET) PROCESSOR --- #
    
    Dense21 = Dense(128, activation='gelu', name='dense128')(other_input)
    Dense22 = Dense(64,  activation='gelu', name='dense64')(Dense21)
    flat_other = Dense22
    
    # --- JET CLASSIFIER --- #
    
    if Model.use_JetPretraining:
        jet_pretrain_model.trainable = False                                      # Freezing the jet pretrain model (i.e. want to use the previously trained weights)
        j_weights = jet_pretrain_model([jet_input,other_input], training=False)   # Putting the inputs into the pretrain model
    else:
        flat_jets =  Flatten(name ='flattened_jets')(jet_input) 
        concat0 = concatenate([other_input, flat_jets], name = 'concat_jets_other')
        PreDense1 = Dense(256, activation='gelu', name = 'dense256_1')(concat0)
        PreDense2 = Dense(256, activation='gelu', name = 'dense256_2')(PreDense1) 
        j_weights = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(PreDense2)
        
    # --- BBBAR CLASSIFIER --- #
    
    if Model.use_bbPretraining:
        bb_pretrain_model.trainable = False                                      # Freezing the bb pretrain model (i.e. want to use the previously trained weights)
        b_weights = bb_pretrain_model([jet_input,other_input], training=False)   # Putting the inputs into the pretrain model
    else:
        bflat_jets =  Flatten(name ='b_flattened_jets')(jet_input) 
        bconcat0 = concatenate([other_input, bflat_jets], name = 'concat_bjets_other')
        bPreDense1 = Dense(256, activation='gelu', name = 'dense256_b1')(bconcat0)
        bPreDense2 = Dense(256, activation='gelu', name = 'dense256_b2')(bPreDense1) 
        b_weights = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_b_sigmoid')(bPreDense2)

    # --- WEIGHTED JET PROCESSOR --- #
    
    Shape_Dot = Reshape((-1,1), name='reshape')(j_weights)
    wjets = Multiply(name='weight_jets')([Shape_Dot, TDDense12])  # Weight the jets from initial jet processor by the weights from jet classifier
    TDDense13 = TimeDistributed(Dense(256, activation='gelu'), name='TDDense256_1')(wjets)
    TDDense14= TimeDistributed(Dense(256, activation='gelu'), name='TDDense256_2')(TDDense13)
    Flat_wjets = Flatten(name='flattened_weighted_jets')(TDDense14)
    
    # --- WEIGHTED BBBAR PROCESSOR --- #
    
    bShape_Dot = Reshape((-1,1), name='reshape_b')(b_weights)
    b_wjets = Multiply(name='weight_bjets')([bShape_Dot, bTDDense13])  # Weight the jets from initial bbbar jet processor by the weights from bbbar classifier
    b_TDDense13 = TimeDistributed(Dense(256, activation='gelu'), name='TDDense256_b1')(b_wjets)
    b_TDDense14= TimeDistributed(Dense(256, activation='gelu'), name='TDDense256_b2')(b_TDDense13)
    b_Flat_wjets = Flatten(name='flattened_weighted_bjets')(b_TDDense14)
    
    # --- FINAL PROCESSOR --- #
    
    # Concatenate the two sides
    concat = concatenate([flat_other, Flat_wjets], name = 'concat_everything')
    concat = LayerNormalization(center=False, scale=False, name='LN_concat')(concat)

    # Get leptonic output
    ldense1 = Dense(256, activation='gelu', name='ldense256')(concat)
    ldrop1 = Dropout(0.1, name='DO_ldense256')(ldense1)
    ldense2 = Dense(128, activation='gelu', name='ldense128')(ldrop1)
    loutput = Dense(Model.lep_shape, name='lep_output')(ldense2)
    
    # Get hadronic (+ttbar) output
    hconcat = concatenate([loutput, concat])
    hdense1 = Dense(256, activation='gelu', name='hdense256')(hconcat)
    hdrop1 = Dropout(0.1, name='DO_hdense256')(hdense1)
    hdense2 = Dense(128, activation='gelu', name='hdense128')(hdrop1)
    houtput = Dense(Model.had_shape+Model.ttbar_shape, name='had_output')(hdense2)

    # Get bbbar output
    bconcat = concatenate([houtput,b_Flat_wjets])
    bdense1 = Dense(256, activation='gelu', name='bdense256')(bconcat)
    bhdrop1 = Dropout(0.1, name='DO_bdense256')(bdense1)
    bdense2 = Dense(128, activation='gelu', name='bdense128')(bhdrop1)
    boutput = Dense(Model.bbbar_shape, name='bbbar_output')(bdense2)

    # Final output
    output = concatenate([houtput, loutput, boutput], name='final_output')
            
    return output