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
from keras import layers
from .objwise import ObjWise
from utils.attention_utils import make_attn_mask

# TODO
class SelfAttentionBlock(layers.Layer):

    def __init__(self, dim_model, num_heads, dim_ffn, dropout = 0.1, **kw):
        super.__init__(**kw)



    def call(self, x, valid):

