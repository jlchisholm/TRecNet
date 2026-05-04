##########################################################################
#                                                                        #
#  TRecNet_ttbb_v5x3.py                                                  #
#  Author: Tommy Lubomirski                                              #
#  Updated: Dec/25                                                       #
##########################################################################

from keras.layers import Dense, Dropout, LayerNormalization, concatenate, Flatten
from keras import backend as K
from keras import ops as kop
import tensorflow as tf

from .blocks.set_encoder import JetSetEncoder
from .blocks.pooling import AttentionPooling, MultiQueryPooling
from .blocks.transformer_blocks import CrossAttentionBlock

def construct_TRecNet_ttbb_v5x3(Model, jet_input, other_input, jet_pretrain_model, bb_pretrain_model, hparams=None):
    hp = dict(hparams or {})
    # Set hyperparams here so its easier to make them visible if this model warants hypertuning
    D_MODEL = hp.get("d_model", 128)
    FFN_DIM = hp.get("ffn_dim", 512)
    N_HEADS = hp.get("n_heads", 8)
    N_BLOCKS = hp.get("n_blocks", 3)
    DROPOUT_P = hp.get("enc_dropout", 0.1)

    POOL_HEADS = hp.get("pool_heads", max(1, D_MODEL // 32))
    POOL_DO = hp.get("pool_dropout", 0.0)
    USE_MULTI_QUERY = hp.get("use_multi_query_pool", True)
    NUM_QUERIES = hp.get("num_queries", 3)

    USE_CONTEXT_TOKEN = hp.get("use_context_token", True)
    CONTEXT_DIM = hp.get("context_dim", None) or D_MODEL

    USE_REL_ATTN_BIAS = hp.get("use_rel_attn_bias", True)
    REL_HIDDEN = hp.get("rel_hidden", 32)
    REL_HEAD_SCALE = hp.get("rel_head_scale", 1.0)
    REL_FEAT_MODE = hp.get("rel_feat_mode", "lite")

    CLS_ACT = hp.get("cls_activ", "gelu")
    CLS_DO  = hp.get("cls_dropout", 0.0)
    jet_cls_mlp  = hp.get("jet_cls_mlp", [128, 128])
    bjet_cls_mlp = hp.get("bjet_cls_mlp", [128, 128])

    OTHER_ACT = hp.get("other_activ", "gelu")
    OTHER_DO  = hp.get("other_dropout", 0.1)
    other_mlp = hp.get("other_mlp", [128, 64])

    LN_CENTER = hp.get("trunk_ln_center", False)
    LN_SCALE  = hp.get("trunk_ln_scale", False)

    FINAL_ACT = hp.get("final_activ", "gelu")
    FINAL_DO  = hp.get("final_dropout", 0.1)
    final_mlp = hp.get("final_mlp", [256, 128, 256])

    HEAD_ACT = hp.get("head_activ", "gelu")
    HEAD_DO  = hp.get("head_dropout", 0.1)
    lep_head = hp.get("lep_head", [128, 64])
    had_head = hp.get("had_head", [128, 64])
    bb_head  = hp.get("bb_head",  [128, 64])

    # turn jets into tokens for (B, N, D_MODEL) with model mask value
    encoder = JetSetEncoder(
        dim_model=D_MODEL,
        dim_ffn=FFN_DIM,
        num_heads=N_HEADS, 
        num_blocks=N_BLOCKS,
        dropout=DROPOUT_P,
        mask_value = Model.mask_value,
        use_rel_attn_bias=USE_REL_ATTN_BIAS,
        rel_hidden=REL_HIDDEN,
        rel_head_scale=REL_HEAD_SCALE,
        rel_feat_mode=REL_FEAT_MODE,
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
        x = concat0
        for i, units in enumerate(jet_cls_mlp, start=1):
            x = Dense(units, activation=CLS_ACT, name=f"dense_jcls_{units}_{i}")(x)
            if CLS_DO and CLS_DO > 0:
                x = Dropout(CLS_DO, name=f"do_jcls_{units}_{i}")(x)
        j_weights = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_sigmoid')(x)
        
    # --- BBBAR CLASSIFIER --- #
    
    if Model.use_bbPretraining:
        bb_pretrain_model.trainable = False                                      # Freezing the bb pretrain model (i.e. want to use the previously trained weights)
        b_weights = bb_pretrain_model([jet_input,other_input], training=False)   # Putting the inputs into the pretrain model
    else:
        bflat_jets =  Flatten(name ='b_flattened_jets')(jet_input) 
        bconcat0 = concatenate([other_input, bflat_jets], name = 'concat_bjets_other')
        x = bconcat0
        for i, units in enumerate(bjet_cls_mlp, start=1):
            x = Dense(units, activation=CLS_ACT, name=f"dense_bcls_{units}_{i}")(x)
            if CLS_DO and CLS_DO > 0:
                x = Dropout(CLS_DO, name=f"do_bcls_{units}_{i}")(x)
        b_weights = Dense(Model.jets_shape[1], activation='sigmoid', name='dense6_b_sigmoid')(x)



    # --- TOKEN GATING (stronger than pooling weights) --- #
    # same thing as in v5x1 where I have to convert to keras tensor
    j_weights = kop.clip(j_weights, 1e-6, 1.0 - 1e-6)
    b_weights = kop.clip(b_weights, 1e-6, 1.0 - 1e-6)
    # gating hparams should really move to top
    j_gate_pow       = hp.get("j_gate_pow", 1.0)
    b_gate_scale     = hp.get("b_gate_scale", 0.5)
    gate_temperature = hp.get("gate_temperature", 1.0)
    gate_floor       = hp.get("gate_floor", 0.20)
    hard_threshold   = hp.get("gate_hard_threshold", None)
    # make the prior and gate the values
    # the prior is a classifier outputs that get scaled and shifted, then passed through sigmoid
    #  to get a gate value between 0 and 1
    # a bit redundant but i had to do it this way
    prior = kop.power(j_weights + K.epsilon(), j_gate_pow) * (1.0 + b_gate_scale * b_weights)
    prior_mean = kop.stop_gradient(kop.mean(prior, axis=1, keepdims=True))
    prior = prior / (prior_mean + K.epsilon())
    prior = kop.sigmoid((prior - 1.0) / (gate_temperature + K.epsilon()))
    gate  = gate_floor + (1.0 - gate_floor) * prior
    jtokens = jtokens * gate[..., None]
    if hard_threshold is not None:
        drop_mask = kop.less(gate, hard_threshold)
        jmask = kop.logical_or(jmask, drop_mask)

    # --- CONTEXT TOKEN CROSS-ATTN--- #
    if USE_CONTEXT_TOKEN:
        ctx = LayerNormalization(name="LN_context_token")(other_input)
        ctx = Dense(CONTEXT_DIM, activation='gelu', name="context_token_dense")(ctx)
        ctx = kop.expand_dims(ctx, axis=1)
        jtokens = CrossAttentionBlock(D_MODEL, N_HEADS, FFN_DIM, DROPOUT_P, name="jet_context_xattn")(jtokens, ctx, jmask)

    # --- POOLING --- #
    if USE_MULTI_QUERY:
        pooled = MultiQueryPooling(
            dim_model=D_MODEL,
            num_heads=POOL_HEADS,
            num_queries=NUM_QUERIES,
            dropout=POOL_DO,
            name="multi_query_pool",
        )(jtokens, jmask)
        had_vec = pooled[:, 0, :]
        lep_vec = pooled[:, 1, :] if NUM_QUERIES > 1 else pooled[:, 0, :]
        bb_vec  = pooled[:, 2, :] if NUM_QUERIES > 2 else pooled[:, 0, :]
    else:
        jet_vec = AttentionPooling(
            dim_model= D_MODEL,
            num_heads= POOL_HEADS,
            dropout = POOL_DO, name='jet_attention_pool')(jtokens, jmask, weights=j_weights) # (B, D_MODEL)
        bb_vec = AttentionPooling(
            dim_model=D_MODEL,
            num_heads=POOL_HEADS,
            dropout=POOL_DO,
            name='bb_attention_pool'
        )(jtokens, jmask, weights=b_weights) # (B, D_MODEL)
        had_vec = jet_vec
        lep_vec = jet_vec

    # --- INITIAL OTHER (LEP+MET) PROCESSOR --- #
    # Im keeping the parameter count similar to v5 but could be expanded
    other_ln = LayerNormalization(name='LN_other')(other_input)
    x = other_ln
    for i, units in enumerate(other_mlp, start=1):
        x = Dense(units, activation=OTHER_ACT, name=f"dense_other_{units}_{i}")(x)
        if OTHER_DO and OTHER_DO > 0 and i == 1:
            x = Dropout(rate = OTHER_DO, name=f"do_other_{units}_{i}")(x)
    context_vec = x

    # create the full event representation vector
    concat = concatenate([had_vec, lep_vec, bb_vec, context_vec], name='event_concat')
    concat = LayerNormalization(center=LN_CENTER, scale = LN_SCALE, name='ln_event_vec')(concat)

    x = concat
    for i, units in enumerate(final_mlp, start=1):
        x = Dense(units, activation=FINAL_ACT, name=f"final_dense_{units}_{i}")(x)
        if FINAL_DO and FINAL_DO > 0:
            x = Dropout(FINAL_DO, name=f"final_do_{units}_{i}")(x)
    concat = x

    # --- FINAL PROCESSOR --- #

    # Get leptonic output
    x = concat
    for i, units in enumerate(lep_head, start=1):
        x = Dense(units, activation=HEAD_ACT, name=f"lep_head_{units}_{i}")(x)
        if HEAD_DO and HEAD_DO > 0 and i == 1:
            x = Dropout(HEAD_DO, name=f"do_lep_{units}_{i}")(x)
    loutput = Dense(Model.lep_shape, name='lep_output')(x)
    
    # Get hadronic (+ttbar) output
    hconcat = concatenate([loutput, concat, had_vec])
    x = hconcat
    for i, units in enumerate(had_head, start=1):
        x = Dense(units, activation=HEAD_ACT, name=f"had_head_{units}_{i}")(x)
        if HEAD_DO and HEAD_DO > 0 and i == 1:
            x = Dropout(HEAD_DO, name=f"do_had_{units}_{i}")(x)
    houtput = Dense(Model.had_shape+Model.ttbar_shape, name='had_output')(x)

    # Get bbbar output
    bconcat = concatenate([houtput, bb_vec])
    x = bconcat
    for i, units in enumerate(bb_head, start=1):
        x = Dense(units, activation=HEAD_ACT, name=f"bb_head_{units}_{i}")(x)
        if HEAD_DO and HEAD_DO > 0 and i == 1:
            x = Dropout(HEAD_DO, name=f"do_bb_{units}_{i}")(x)
    boutput = Dense(Model.bbbar_shape, name='bbbar_output')(x)

    # Final output
    output = concatenate([houtput, loutput, boutput], name='final_output')
            
    return output
