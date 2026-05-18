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

from keras.layers import Flatten, Dense, concatenate, Masking, TimeDistributed, Reshape, Multiply

def construct_TRecNet_ttbb_v5(Model,jet_input, 
                              other_input,
                              jet_pretrain_model,
                              bb_pretrain_model,
                              hparams = None):
    

    '''
    hparams keys (all optional):
      jet_td: [128, 64]
      bjet_td: [128, 128, 64]
      other_mlp: [128, 64]
      jet_cls_mlp: [256, 256]
      bjet_cls_mlp: [256, 256]
      weighted_td: [256, 256]
      weighted_b_td: [256, 256]
      lep_head: [256, 128]
      had_head: [256, 128]
      bb_head: [256, 128]
      activ: "relu"
      cls_activ: "relu"
    '''
    
    hp = dict(hparams or {})
    # --- INITIAL JET PROCESSOR --- #
    activ = hp.get("activ", "relu")
    cls_activ = hp.get("cls_activ", activ)

    jet_td = hp.get("jet_td", [128, 64])
    bjet_td = hp.get("bjet_td", [128, 128, 64])
    other_mlp = hp.get("other_mlp", [128, 64])

    jet_cls_mlp = hp.get("jet_cls_mlp", [256, 256])
    bjet_cls_mlp = hp.get("bjet_cls_mlp", [256, 256])

    weighted_td = hp.get("weighted_td", [256, 256])
    weighted_b_td = hp.get("weighted_b_td", [256, 256])

    lep_head = hp.get("lep_head", [256, 128])
    had_head = hp.get("had_head", [256, 128])
    bb_head  = hp.get("bb_head",  [256, 128])
                                 

    Mask = Masking(Model.mask_value, name='masking_jets')(jet_input)
    Maskshape = Reshape((Model.jets_shape[1], Model.jets_shape[2]), name='reshape_masked_jets')(Mask)
    TDDense11 = TimeDistributed(Dense(128, activation='relu'), name='TDDense128')(Maskshape)
    TDDense12 = TimeDistributed(Dense(64, activation='relu'), name='TDDense64')(TDDense11)
    
    # --- INITIAL BBBAR JET PROCESSOR --- #
    
    bMask = Masking(Model.mask_value, name='b_masking_jets')(jet_input)
    bMaskshape = Reshape((Model.jets_shape[1], Model.jets_shape[2]), name='b_reshape_masked_jets')(bMask)
    bTDDense11 = TimeDistributed(Dense(128, activation='relu'), name='TDDense128_b1')(bMaskshape)
    bTDDense12 = TimeDistributed(Dense(128, activation='relu'), name='TDDense128_b2')(bTDDense11)
    bTDDense13 = TimeDistributed(Dense(64, activation='relu'), name='TDDense128_b3')(bTDDense12)
    
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
        
    # --- BBBAR CLASSIFIER --- #
    
    if Model.use_bbPretraining:
        bb_pretrain_model.trainable = False                                      # Freezing the bb pretrain model (i.e. want to use the previously trained weights)
        b_weights = bb_pretrain_model([jet_input,other_input], training=False)   # Putting the inputs into the pretrain model
    else:
        bflat_jets =  Flatten(name ='b_flattened_jets')(jet_input) 
        bconcat0 = concatenate([other_input, bflat_jets], name = 'concat_bjets_other')
        bPreDense1 = Dense(256, activation='relu', name = 'dense256_b1')(bconcat0)
        bPreDense2 = Dense(256, activation='relu', name = 'dense256_b2')(bPreDense1) 
        b_weights = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_b_sigmoid')(bPreDense2)

    # --- WEIGHTED JET PROCESSOR --- #
    
    Shape_Dot = Reshape((-1,1), name='reshape')(j_weights)
    wjets = Multiply(name='weight_jets')([Shape_Dot, TDDense12])  # Weight the jets from initial jet processor by the weights from jet classifier
    TDDense13 = TimeDistributed(Dense(256, activation='relu'), name='TDDense256_1')(wjets)
    TDDense14= TimeDistributed(Dense(256, activation='relu'), name='TDDense256_2')(TDDense13)
    Flat_wjets = Flatten(name='flattened_weighted_jets')(TDDense14)
    
    # --- WEIGHTED BBBAR PROCESSOR --- #
    
    bShape_Dot = Reshape((-1,1), name='reshape_b')(b_weights)
    b_wjets = Multiply(name='weight_bjets')([bShape_Dot, bTDDense13])  # Weight the jets from initial bbbar jet processor by the weights from bbbar classifier
    b_TDDense13 = TimeDistributed(Dense(256, activation='relu'), name='TDDense256_b1')(b_wjets)
    b_TDDense14= TimeDistributed(Dense(256, activation='relu'), name='TDDense256_b2')(b_TDDense13)
    b_Flat_wjets = Flatten(name='flattened_weighted_bjets')(b_TDDense14)
    
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
    
    # Get bbbar output
    bconcat = concatenate([houtput,b_Flat_wjets])
    bdense1 = Dense(256, activation='relu', name='bdense256')(bconcat)
    bdense2 = Dense(128, activation='relu', name='bdense128')(bdense1)
    boutput = Dense(Model.bbbar_shape, name='bbbar_output')(bdense2)

    # Final output
    output = concatenate([loutput, houtput, boutput], name='final_output')
            
    return output