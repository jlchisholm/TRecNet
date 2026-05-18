##########################################################################
#                                                                        #
#  TRecNet_ttbb_v5x1_clf.py                                                  #
#  Author: Tommy Lubomirski                                              #
#  (based on TRecNet_ttbb_v5.py by Jenna Chisholm)                       #         
#  Updated: Oct.4/25                                                     #
#                                                                        #
#  Function to construct architecture for version v5x1 of TRecNet.
#  This model implements architecture developed for the Self-attention 
#  Jet Assignment networkk described in the paper:
# https://arxiv.org/abs/2012.03542
#
# Lee, J. S., Park, I., Watson, I. J., & Yang, S. (2024).
# Zero-permutation jet-parton assignment using a self-attention network.
# Journal of the Korean Physical Society, 84(6), 427–438.
# https://doi.org/10.1007/s40042-024-01037-3 
# 
# This model replaces the TimeDistributed-Dense jet stacks with a shared,
# mask aware Transformer set encoder with attention pooling (SAJA paper)
# 
# The function signature remains the same as of v5 (inputs and outpus are
# the same) to ensure compatibility with the TRecNet existing
# infrastructure
##########################################################################
from keras.layers import Dense, Dropout, LayerNormalization, concatenate, Flatten, Reshape
from .blocks.set_encoder import JetSetEncoder
from .blocks.pooling import AttentionPooling
from keras import backend as K
from keras import ops as kop
import tensorflow as tf

def construct_TRecNet_ttbb_v5x1_clf(Model,jet_input, other_input,jet_pretrain_model,bb_pretrain_model):
    
    # Set hyperparams here so its easier to make them visible if this model warants hypertuning
    # NOTE  arch hyperparam optim is not set up for this model
    D_MODEL = 128
    FFN_DIM = 512
    N_HEADS = 8
    N_BLOCKS = 3
    DROPOUT_P = 0.1


    # turn jets into tokens for (B, N, D_MODEL) with model mask value
    encoder = JetSetEncoder(
        dim_model=D_MODEL,
        dim_ffn=FFN_DIM,
        num_heads=N_HEADS, 
        num_blocks=N_BLOCKS,
        dropout=DROPOUT_P,
        mask_value = Model.mask_value,
        name = 'jet_encoder')

    # get the tokens an the mask (B,N,D), (D,N)
    jtokens, jmask = encoder(jet_input) 


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

    # we want to use the classifier weights to gate the jet tokens in the attention pooling
    # expand the weights to match the token dimensions
    # I was getting an error because I did a tensor flow op on a keras tensor, so i just passed
    # them through a clip layer to make them a keras tensor again,
    j_weights = kop.clip(j_weights, 1e-6, 1.0 - 1e-6)  # (B,N)
    b_weights = kop.clip(b_weights, 1e-6, 1.0 - 1e-6)  # (B,N)

    # these hyperparams are currently fixed but im making them visible in case we want to tune them later
    j_gate_pow       = getattr(Model, 'j_gate_pow', 1.0)   # sharpen/flatten jet clf
    b_gate_scale     = getattr(Model, 'b_gate_scale', 0.5)  # how much b-ness nudges
    gate_temperature = getattr(Model, 'gate_temperature', 1.0)  # softness of sigmoid
    gate_floor       = getattr(Model, 'gate_floor', 0.20)   # min gate in [0,1]
    hard_threshold   = getattr(Model, 'gate_hard_threshold', None)  # e.g., 0.03 or None

    # combine priors j_weights with a power and b_weights with a scale
    prior = kop.power(j_weights + K.epsilon(), j_gate_pow) * (1.0 + b_gate_scale * b_weights)

    # mean-normalize across jets so average scaling stays near 1
    prior_mean = kop.stop_gradient(kop.mean(prior, axis=1, keepdims=True))
    prior = prior / (prior_mean + K.epsilon())

    # squish around 1 and map to [gate_floor, 1]
    prior = kop.sigmoid((prior - 1.0) / (gate_temperature + K.epsilon()))
    gate  = gate_floor + (1.0 - gate_floor) * prior
    # add this for now to stop gradient flow through gate
    # but remove later to allow end-to-end training
    #gate  = kop.stop_gradient(gate) 
    #apply gate to tokens (broadcast along D)
    jtokens = jtokens * gate[..., None]   # (B, N, D_MODEL)
    # hard drop: integrate gate into mask for very low-score jets
    if hard_threshold is not None:
        # drop_mask: True where we want to mask/drop
        drop_mask = kop.less(gate, hard_threshold)  # bool (B,N)

        jmask = kop.logical_or(jmask, drop_mask)



    # pool the tokens and get an event(jets) representation from the learned query
    # attention pooling
    jet_vec = AttentionPooling(
        dim_model= D_MODEL,
        num_heads= D_MODEL // 32,
        dropout = 0.0, name='jet_attention_pool')(jtokens,jmask) # (B, D_MODEL)

    # --- INITIAL OTHER (LEP+MET) PROCESSOR --- #
    # Im keeping the parameter count similar to v5 but could be expanded
    other_ln = LayerNormalization(name='LN_other')(other_input)
    Dense21 = Dense(128, activation='gelu', name='dense128')(other_ln)
    other_do = Dropout(rate = 0.1, name='do_other_128')(Dense21)
    Dense22 = Dense(64,  activation='gelu', name='dense64')(other_do)
    context_vec = Dense22 #(B, 64)
    
    # create the full event representation vector
    concat = concatenate([jet_vec, context_vec], name = 'event_concat') #(B, 64 + D_Model)
    concat = LayerNormalization(center=False, scale = False, name='ln_event_vec')(concat)

    # --- FINAL PROCESSOR --- #

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
    bconcat = concatenate([houtput,jet_vec])
    bdense1 = Dense(256, activation='gelu', name='bdense256')(bconcat)
    bhdrop1 = Dropout(0.1, name='DO_bdense256')(bdense1)
    bdense2 = Dense(128, activation='gelu', name='bdense128')(bhdrop1)
    boutput = Dense(Model.bbbar_shape, name='bbbar_output')(bdense2)

    # Final output
    output = concatenate([loutput, houtput, boutput], name='final_output')
            
    return output



