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
from .attention_utils import make_attn_mask

@saving.register_keras_serializable(package="Models.blocks", name="AttentionPooling")
class AttentionPooling(layers.Layer):

    ''' create a single **learned**  query vector, tile it for each batch, an run 
        one to many attention on it. This outputs a single pooled event vector (B,D)'''
    def __init__(self, dim_model, num_heads:int = 4, dropout = 0.01, **kw):
        super().__init__(**kw)
        if num_heads <= 0:
            raise Exception("num_heads not valid")

        #query
        # trainable vector, the documentation page says that add_weight is not trainable but 
        # but it actually is trainable = True by default so idk
        self.q = self.add_weight(name = 'pool_query',
         shape = (1,1, dim_model), initializer = 'glorot_uniform') 
         # xaviar initialization is called glorot in keras apparently
        
        self.mha = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = dim_model // num_heads, dropout = dropout
            )
        
    def call(self, tokens, valid, weights=None):
        B = tf.shape(tokens)[0]
        q = tf.tile(self.q, [B, 1, 1])  # (B, 1, D)

        # padding mask ONLY (boolean)
        attn_mask = make_attn_mask(valid)  # expects valid in {0,1} for padding

        # soft weights applied to token VALUES (not the mask)
        if weights is not None:
            w = tf.clip_by_value(weights, 0.0, 1.0)                 # (B, N)
            tokens = tokens * tf.expand_dims(w, axis=-1)            # (B, N, D)

        pooled = self.mha(query=q, value=tokens, key=tokens, attention_mask=attn_mask)
        return tf.squeeze(pooled, axis=1)

@saving.register_keras_serializable(package="Models.blocks", name="MultiQueryPooling")
class MultiQueryPooling(layers.Layer):
    '''Multiple learned queries to pool set tokens into Q vectors (B,Q,D).
    part of this code was written by GenAI'''
    def __init__(self, dim_model, num_heads:int = 4, num_queries:int = 3, dropout = 0.01, **kw):
        super().__init__(**kw)
        if num_heads <= 0:
            raise Exception("num_heads not valid")
        if num_queries <= 0:
            raise Exception("num_queries not valid")

        self.num_queries = num_queries
        self.q = self.add_weight(
            name='pool_queries',
            shape=(1, num_queries, dim_model),
            initializer='glorot_uniform'
        )
        self.mha = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = dim_model // num_heads, dropout = dropout
        )

    def call(self, tokens, valid, weights=None):
        B = tf.shape(tokens)[0]
        q = tf.tile(self.q, [B, 1, 1])  # (B, Q, D)

        attn_mask = make_attn_mask(valid)

        if weights is not None:
            w = tf.clip_by_value(weights, 0.0, 1.0)
            tokens = tokens * tf.expand_dims(w, axis=-1)

        pooled = self.mha(query=q, value=tokens, key=tokens, attention_mask=attn_mask)
        return pooled  # (B, Q, D)

        




