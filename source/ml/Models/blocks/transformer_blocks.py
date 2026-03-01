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
from .objwise import ObjWise
from .attention_utils import make_attn_mask


@saving.register_keras_serializable(package="Models.blocks", name="SelfAttentionBlock")
class SelfAttentionBlock(layers.Layer):
    ''' this is a transformer block over jet (tokens) 
    it has an MHA with a mask over tokens with a residual connection to prevent gradients dying
    it passe through an object wise layer norm so only valid tokens are touched
    and then through an FFN'''
    def __init__(self, dim_model, num_heads, dim_ffn, dropout = 0.1, **kw):
        super().__init__(**kw)
        self.mha = layers.MultiHeadAttention(num_heads = num_heads, key_dim = dim_model // num_heads, dropout=dropout)
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.head_dim = dim_model // num_heads

        # qkv projections for relation-aware attention
        self.q_proj = layers.Dense(dim_model, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(dim_model, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(dim_model, use_bias=False, name="v_proj")
        self.out_proj = layers.Dense(dim_model, use_bias=False, name="out_proj")
        # attention dropout
        self.do_attn = layers.Dropout(rate = dropout)
        # layerNorm attention
        self.ln_attn = ObjWise(layers.LayerNormalization(epsilon = 1e-6))

        # feed forward net: objwise so only valid tokens are processed
        self.ffn = ObjWise([
            layers.Dense(dim_ffn, activation = 'gelu'),
            layers.Dropout(rate = dropout),
            layers.Dense(dim_model, activation = 'gelu')
        ])
        # ffn dropout
        self.do_ffn = layers.Dropout(rate = dropout)
        # ffn Layer Norm
        self.ln_ffn = ObjWise(layers.LayerNormalization(epsilon = 1e-6))

    def _split_heads(self, x):
        # x: (B, N, D) -> (B, H, N, Hd)
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _combine_heads(self, x):
        # x: (B, H, N, Hd) -> (B, N, D)
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.dim_model])

    def _rel_attn(self, x, valid, attn_bias):
        # x: (B, N, D), attn_bias: (B, H, N, N)
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        dk = tf.cast(self.head_dim, x.dtype)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        if attn_bias is not None:
            scores = scores + attn_bias

        # apply padding mask on keys
        keep = tf.cast(valid > 0.5, scores.dtype)  # (B, N)
        key_mask = tf.expand_dims(tf.expand_dims(keep, 1), 2)  # (B,1,1,N)
        scores = scores + (1.0 - key_mask) * tf.cast(-1e9, scores.dtype)

        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, v)
        out = self._combine_heads(out)
        return self.out_proj(out)

    def call(self, x, valid, attn_bias=None):
        ''' x: (B, N, D); valid: (B,N)'''

        attn_mask = make_attn_mask(valid) # this is (B, 1, N)
        # self atten so all values are x including query
        x_ln  = self.ln_attn(x, valid)
        if attn_bias is None:
            y = self.mha(query = x_ln, value = x_ln, key = x_ln, attention_mask = attn_mask) # (B, N, D))
        else:
            y = self._rel_attn(x_ln, valid, attn_bias)
        x = x + self.do_attn(y)

        x_ln = self.ln_ffn(x, valid)
        y = self.ffn(x_ln, valid)
        x = x + self.do_ffn(y)
        return x

@saving.register_keras_serializable(package="Models.blocks", name="CrossAttentionBlock")
class CrossAttentionBlock(layers.Layer):
    ''' cross-attention block: queries from tokens, keys/values from context token '''
    def __init__(self, dim_model, num_heads, dim_ffn, dropout = 0.1, **kw):
        super().__init__(**kw)
        self.mha = layers.MultiHeadAttention(num_heads = num_heads, key_dim = dim_model // num_heads, dropout=dropout)
        self.do_attn = layers.Dropout(rate = dropout)
        self.ln_attn = ObjWise(layers.LayerNormalization(epsilon = 1e-6))

        self.ffn = ObjWise([
            layers.Dense(dim_ffn, activation = 'gelu'),
            layers.Dropout(rate = dropout),
            layers.Dense(dim_model, activation = 'gelu')
        ])
        self.do_ffn = layers.Dropout(rate = dropout)
        self.ln_ffn = ObjWise(layers.LayerNormalization(epsilon = 1e-6))

    def call(self, x, ctx, valid):
        # x: (B,N,D), ctx: (B,1,D)
        x_ln = self.ln_attn(x, valid)
        y = self.mha(query=x_ln, value=ctx, key=ctx)  # (B,N,D)
        x = x + self.do_attn(y)

        x_ln = self.ln_ffn(x, valid)
        y = self.ffn(x_ln, valid)
        x = x + self.do_ffn(y)
        return x

@saving.register_keras_serializable(package="Models.blocks", name="ObjFFNBottom")
class ObjFFNBottom(layers.Layer):
    ''' for per jet embedding
    The ObjFFNBottom goes per jet F -> D
    '''
    def __init__(self, dim_ffn, dim_model, dropout = 0.1):
        super().__init__()
        self.net = ObjWise([
            layers.Dense(dim_ffn, activation = 'gelu'),
            layers.Dropout(rate = dropout),
            layers.Dense(dim_model, activation = 'gelu')
        ])

    def call(self, x, valid):
        return self.net(x, valid)

@saving.register_keras_serializable(package="Models.blocks", name="ObjFFNTop")
class ObjFFNTop(layers.Layer):
    ''' for per jet embedding
    the ObjFFNTOP goes per jet  D -> C this is here for future use, not used in v5x1
    '''

    # do i need to pass dim_model?
    def __init__(self, dim_model, dim_ffn, dim_out, dropout = 0.1):
        super().__init__()
        self.net = ObjWise([
            layers.Dense(dim_ffn, activation = 'gelu'),
            layers.Dropout(rate = dropout),
            # output is linear, can attach a sigmoid but raw embedding might be the move rn
            layers.Dense(dim_out) 

        ])

    def call(self, x, valid):
        return self.net(x, valid)




