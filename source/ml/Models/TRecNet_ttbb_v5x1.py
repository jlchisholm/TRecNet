##########################################################################
#                                                                        #
#  TRecNet_ttbb_v5x1.py                                                  #
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

def construct_TRecNet_ttbb_v5x1(Model,jet_input, other_input,
                                jet_pretrain_model,bb_pretrain_model,
                                hparams = None):
    
    hp = dict(hparams or {})

    D_MODEL   = hp.get("d_model", 128)
    FFN_DIM   = hp.get("ffn_dim", 512)
    N_HEADS   = hp.get("n_heads", 8)
    N_BLOCKS  = hp.get("n_blocks", 3)
    ENC_DO    = hp.get("enc_dropout", 0.1)

    POOL_HEADS = hp.get("pool_heads", max(1, D_MODEL // 32))
    POOL_DO    = hp.get("pool_dropout", 0.01)

    OTHER_ACT = hp.get("other_activ", "gelu")
    OTHER_DO  = hp.get("other_dropout", 0.1)
    other_mlp = hp.get("other_mlp", [128, 64])

    HEAD_ACT = hp.get("head_activ", "gelu")
    HEAD_DO  = hp.get("head_dropout", 0.1)
    lep_head = hp.get("lep_head", [256, 128])
    had_head = hp.get("had_head", [256, 128])
    bb_head  = hp.get("bb_head",  [256, 128])

    LN_CENTER = hp.get("trunk_ln_center", False)
    LN_SCALE  = hp.get("trunk_ln_scale", False)


    # turn jets into tokens for (B, N, D_MODEL) with model mask value
    encoder = JetSetEncoder(
        dim_model=D_MODEL,
        dim_ffn=FFN_DIM,
        num_heads=N_HEADS, 
        num_blocks=N_BLOCKS,
        dropout=ENC_DO,
        mask_value = Model.mask_value,
        name = 'jet_encoder')

    # get the tokens an the mask (B,N,D), (D,N)
    jtokens, jmask = encoder(jet_input) 


    # pool the tokens and get an event(jets) representation from the learned query
    # attention pooling
    jet_vec = AttentionPooling(
        dim_model= D_MODEL,
        num_heads= POOL_HEADS,
        dropout = POOL_DO, name='jet_attention_pool')(jtokens,jmask) # (B, D_MODEL)

    # --- INITIAL OTHER (LEP+MET) PROCESSOR --- #
    # Im keeping the parameter count similar to v5 but could be expanded
    other_ln = LayerNormalization(name='LN_other')(other_input)
    Dense21 = Dense(other_mlp[0], activation=OTHER_ACT, name='dense128')(other_ln)
    other_do = Dropout(rate = OTHER_DO, name='do_other_128')(Dense21)
    Dense22 = Dense(other_mlp[1],  activation=OTHER_ACT, name='dense64')(other_do)
    context_vec = Dense22 #(B, 64)
    
    # create the full event representation vector
    concat = concatenate([jet_vec, context_vec], name = 'event_concat') #(B, 64 + D_Model)
    concat = LayerNormalization(center=LN_CENTER, scale = LN_SCALE, name='ln_event_vec')(concat)

    # --- FINAL PROCESSOR --- #

    # Get leptonic output
    ldense1 = Dense(lep_head[0], activation=HEAD_ACT, name='ldense256')(concat)
    ldrop1 = Dropout(rate=HEAD_DO, name='DO_ldense256')(ldense1)
    ldense2 = Dense(lep_head[1], activation=HEAD_ACT, name='ldense128')(ldrop1)
    loutput = Dense(Model.lep_shape, name='lep_output')(ldense2)
    
    # Get hadronic (+ttbar) output
    hconcat = concatenate([loutput, concat])
    hdense1 = Dense(had_head[0], activation=HEAD_ACT, name='hdense256')(hconcat)
    hdrop1 = Dropout(rate=HEAD_DO, name='DO_hdense256')(hdense1)
    hdense2 = Dense(had_head[1], activation=HEAD_ACT, name='hdense128')(hdrop1)
    houtput = Dense(Model.had_shape+Model.ttbar_shape, name='had_output')(hdense2)

    # Get bbbar output
    bconcat = concatenate([houtput,jet_vec])
    bdense1 = Dense(bb_head[0], activation=HEAD_ACT, name='bdense256')(bconcat)
    bhdrop1 = Dropout(rate=HEAD_DO, name='DO_bdense256')(bdense1)
    bdense2 = Dense(bb_head[1], activation=HEAD_ACT, name='bdense128')(bhdrop1)
    boutput = Dense(Model.bbbar_shape, name='bbbar_output')(bdense2)

    # Final output
    output = concatenate([houtput, loutput, boutput], name='final_output')
            
    return output



