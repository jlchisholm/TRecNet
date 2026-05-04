# The following code is inspired by the SAJA paper:
# https://arxiv.org/abs/2012.03542


# Lee, J. S., Park, I., Watson, I. J., & Yang, S. (2024).
# Zero-permutation jet-parton assignment using a self-attention network.
# Journal of the Korean Physical Society, 84(6), 427–438.
# https://doi.org/10.1007/s40042-024-01037-3 

# The original codebase is written in PyTorch and can be found here: https://github.com/CPLUOS/SaJa
# The following code is a Tensorflow/Keras adaptation of the original PyTorch code

# Adapted by tommy lubomirski

import tensorflow as tf
from keras import layers, saving
from .attention_utils import build_jet_mask, apply_pad_zeros, build_pairwise_jet_features
from .transformer_blocks import ObjFFNBottom, SelfAttentionBlock

@saving.register_keras_serializable(package="Models.blocks", name="JetSetEncoder")
class JetSetEncoder(layers.Layer):
    ''' mask the jets and zero pad,
     then pass it through a per jet MLP to make features the model dim D
     then make blocks of HA and FFN with residuals and the attention mask
     the output the jet tokens '''
    def __init__(
        self,
        dim_model=128,
        dim_ffn=512,
        num_heads=8,
        num_blocks=3,
        dropout=0.1,
        mask_value=None,
        use_rel_attn_bias=False,
        rel_hidden=32,
        rel_head_scale=1.0,
        rel_feat_mode="lite",
        **kw,
    ):

        super().__init__(**kw)

        self.dim_model = dim_model
        self.bottom = ObjFFNBottom(dim_ffn = dim_ffn, dim_model = dim_model, dropout = dropout) # dim_in is set at call()
        self.mask_value = mask_value
        self.use_rel_attn_bias = use_rel_attn_bias
        self.rel_head_scale = rel_head_scale
        self.rel_feat_mode = rel_feat_mode
        # use a list comprehention to populate blocks
        self.blocks = [SelfAttentionBlock(dim_model, num_heads, dim_ffn, dropout) for _ in range(num_blocks)]
        self.num_heads = num_heads
        if self.use_rel_attn_bias:
            self.rel_dense1 = layers.Dense(rel_hidden, activation='gelu', name="rel_dense1")
            self.rel_dense2 = layers.Dense(num_heads, activation=None, name="rel_dense2")

    def call(self, jets):
        ''' takes jets (B,N,F)
            returns: tokens (B, N, D) where D is the latent dimention
            valid (B, N)
        '''

        valid = build_jet_mask(jets, self.mask_value) #(B,N)
        x = apply_pad_zeros(jets, valid) # (B,N,F)
        # keras will infer the input feature dimentions in compile
        
        # get tokens
        tokens = self.bottom(x, valid)
        # relation-aware attention bias
        # this is not used in the thesis, only applies to v5x3 which is not in the thesis, but is in the codebase for future work
        rel_bias = None
        if self.use_rel_attn_bias:
            pair_feats, pair_mask = build_pairwise_jet_features(jets, valid, mode=self.rel_feat_mode)
            rel = self.rel_dense2(self.rel_dense1(pair_feats))  # (B, N, N, H)
            rel = rel * pair_mask[..., None]
            rel_bias = tf.transpose(rel, [0, 3, 1, 2]) * self.rel_head_scale  # (B, H, N, N)

        # run through self attention blocks
        for block in self.blocks:
            tokens = block(tokens, valid, attn_bias=rel_bias)
        return tokens, valid


        
