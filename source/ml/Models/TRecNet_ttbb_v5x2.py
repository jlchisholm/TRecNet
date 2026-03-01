##########################################################################
#                                                                        #
#  TRecNet_ttbb_v5x2.py                                                  #
#  Author: Tommy Lubomirski                                              #
#  Updated: Dec/25                                                       #
##########################################################################

from keras.layers import Dense, Dropout, LayerNormalization, concatenate, Flatten
from keras import backend as K
from keras import ops as kop
import tensorflow as tf

from .blocks.set_encoder import JetSetEncoder
from .blocks.pooling import AttentionPooling


def construct_TRecNet_ttbb_v5x2(
    Model,
    jet_input,
    other_input,
    jet_pretrain_model,
    bb_pretrain_model,
    hparams=None,
):
    """
    hparams keys (all optional):
      d_model=256, ffn_dim=1024, n_heads=16, n_blocks=4, enc_dropout=0.1
      pool_heads=None (default d_model//32), pool_dropout=0.0
      use_multi_query_pool=True, num_queries=3
      use_context_token=True, context_dim=None (default d_model)
      use_rel_attn_bias=True, rel_hidden=32, rel_head_scale=1.0

      jet_cls_mlp=[256,256], bjet_cls_mlp=[256,256], cls_dropout=0.0, cls_activ="gelu"
      other_mlp=[128,64], other_dropout=0.1, other_activ="gelu"

      trunk_ln_center=False, trunk_ln_scale=False

      final_mlp=[512,256,128,256,512], final_dropout=0.1, final_activ="gelu"

      lep_head=[256,128], had_head=[256,128], bb_head=[256,128], head_dropout=0.1, head_activ="gelu"
    """

    # check if hparams was passed
    hp = dict(hparams or {})

    # default hyperparameters (can be overridden by hparams)
    D_MODEL   = hp.get("d_model", 256)
    FFN_DIM   = hp.get("ffn_dim", 1024)
    N_HEADS   = hp.get("n_heads", 16)
    N_BLOCKS  = hp.get("n_blocks", 4)
    ENC_DO    = hp.get("enc_dropout", 0.1)

    POOL_HEADS = hp.get("pool_heads", max(1, D_MODEL // 32))
    POOL_DO    = hp.get("pool_dropout", 0.0)

    USE_REL_ATTN_BIAS = hp.get("use_rel_attn_bias", True)
    REL_HIDDEN = hp.get("rel_hidden", 32)
    REL_HEAD_SCALE = hp.get("rel_head_scale", 1.0)

    CLS_ACT = hp.get("cls_activ", "gelu")
    CLS_DO  = hp.get("cls_dropout", 0.0)
    jet_cls_mlp  = hp.get("jet_cls_mlp", [256, 256])
    bjet_cls_mlp = hp.get("bjet_cls_mlp", [256, 256])

    OTHER_ACT = hp.get("other_activ", "gelu")
    OTHER_DO  = hp.get("other_dropout", 0.1)
    other_mlp = hp.get("other_mlp", [128, 64])

    LN_CENTER = hp.get("trunk_ln_center", False)
    LN_SCALE  = hp.get("trunk_ln_scale", False)

    FINAL_ACT = hp.get("final_activ", "gelu")
    FINAL_DO  = hp.get("final_dropout", 0.1)
    final_mlp = hp.get("final_mlp", [512, 256, 128, 256, 512])

    HEAD_ACT = hp.get("head_activ", "gelu")
    HEAD_DO  = hp.get("head_dropout", 0.1)
    lep_head = hp.get("lep_head", [256, 128])
    had_head = hp.get("had_head", [256, 128])
    bb_head  = hp.get("bb_head",  [256, 128])

    # --- JET TOKENS --- #
    encoder = JetSetEncoder(
        dim_model=D_MODEL,
        dim_ffn=FFN_DIM,
        num_heads=N_HEADS,
        num_blocks=N_BLOCKS,
        dropout=ENC_DO,
        mask_value=Model.mask_value,
        use_rel_attn_bias=USE_REL_ATTN_BIAS,
        rel_hidden=REL_HIDDEN,
        rel_head_scale=REL_HEAD_SCALE,
        name="jet_encoder",
    )
    # get the tokens an the mask (B,N,D), (D,N)
    jtokens, jmask = encoder(jet_input)

    # --- CLASSIFIERS --- #
    if Model.use_JetPretraining:
        jet_pretrain_model.trainable = False
        j_weights = jet_pretrain_model([jet_input, other_input], training=False)
    else:
        flat_jets = Flatten(name="flattened_jets")(jet_input)
        concat0 = concatenate([other_input, flat_jets], name="concat_jets_other")
        x = concat0
        for i, units in enumerate(jet_cls_mlp, start=1):
            x = Dense(units, activation=CLS_ACT, name=f"dense_jcls_{units}_{i}")(x)
            if CLS_DO and CLS_DO > 0:
                x = Dropout(CLS_DO, name=f"do_jcls_{units}_{i}")(x)
        j_weights = Dense(Model.jets_shape[1], activation="sigmoid", name="jets_sigmoid")(x)

    if Model.use_bbPretraining:
        bb_pretrain_model.trainable = False
        b_weights = bb_pretrain_model([jet_input, other_input], training=False)
    else:
        bflat_jets = Flatten(name="b_flattened_jets")(jet_input)
        bconcat0 = concatenate([other_input, bflat_jets], name="concat_bjets_other")
        x = bconcat0
        for i, units in enumerate(bjet_cls_mlp, start=1):
            x = Dense(units, activation=CLS_ACT, name=f"dense_bcls_{units}_{i}")(x)
            if CLS_DO and CLS_DO > 0:
                x = Dropout(CLS_DO, name=f"do_bcls_{units}_{i}")(x)
        b_weights = Dense(Model.jets_shape[1], activation="sigmoid", name="bjets_sigmoid")(x)

    # --- TOKEN GATING --- #
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


    # --- POOLING --- #
    # pool the tokens
    jet_vec = AttentionPooling(
        dim_model=D_MODEL,
        num_heads=POOL_HEADS,
        dropout=POOL_DO,
        name="jet_attention_pool",
    )(jtokens, jmask)
    had_vec = jet_vec
    lep_vec = jet_vec
    bb_vec = jet_vec

    # --- OTHER CONTEXT --- #
    x = LayerNormalization(name="LN_other")(other_input)
    for i, units in enumerate(other_mlp, start=1):
        x = Dense(units, activation=OTHER_ACT, name=f"dense_other_{units}_{i}")(x)
        if OTHER_DO and OTHER_DO > 0 and i == 1:
            x = Dropout(OTHER_DO, name=f"do_other_{units}_{i}")(x)
    context_vec = x

    # --- TRUNK + FINAL MLP --- #
    concat = concatenate([had_vec, lep_vec, bb_vec, context_vec], name="event_concat")
    concat = LayerNormalization(center=LN_CENTER, scale=LN_SCALE, name="ln_event_vec")(concat)

    x = concat
    for i, units in enumerate(final_mlp, start=1):
        x = Dense(units, activation=FINAL_ACT, name=f"final_dense_{units}_{i}")(x)
        if FINAL_DO and FINAL_DO > 0:
            x = Dropout(FINAL_DO, name=f"final_do_{units}_{i}")(x)
    concat = x

    # --- HEADS --- #
    x = concat
    for i, units in enumerate(lep_head, start=1):
        x = Dense(units, activation=HEAD_ACT, name=f"lep_head_{units}_{i}")(x)
        if HEAD_DO and HEAD_DO > 0 and i == 1:
            x = Dropout(HEAD_DO, name=f"do_lep_{units}_{i}")(x)
    loutput = Dense(Model.lep_shape, name="lep_output")(x)

    hconcat = concatenate([loutput, concat, had_vec], name="concat_had")
    x = hconcat
    for i, units in enumerate(had_head, start=1):
        x = Dense(units, activation=HEAD_ACT, name=f"had_head_{units}_{i}")(x)
        if HEAD_DO and HEAD_DO > 0 and i == 1:
            x = Dropout(HEAD_DO, name=f"do_had_{units}_{i}")(x)
    houtput = Dense(Model.had_shape + Model.ttbar_shape, name="had_output")(x)

    bconcat = concatenate([houtput, bb_vec], name="concat_bb")
    x = bconcat
    for i, units in enumerate(bb_head, start=1):
        x = Dense(units, activation=HEAD_ACT, name=f"bb_head_{units}_{i}")(x)
        if HEAD_DO and HEAD_DO > 0 and i == 1:
            x = Dropout(HEAD_DO, name=f"do_bb_{units}_{i}")(x)
    boutput = Dense(Model.bbbar_shape, name="bbbar_output")(x)

    output = concatenate([houtput, loutput, boutput], name="final_output")
    return output
