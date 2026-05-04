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
from keras import layers, Sequential, saving

@saving.register_keras_serializable(package="Models.blocks", name="ObjWise")
class ObjWise(layers.Layer):
    ''' apply a sublayer only on valid tokens(jets) 
    and scatter back to the original shape'''
    def __init__(self, sublayer, **kwargs):
        super().__init__(**kwargs)
        # allow passing both a list or sublayer (for blocks)
        if isinstance(sublayer, (list, tuple)):
            self.sublayer = Sequential(sublayer)
        else:
            self.sublayer = sublayer
    
    def call(self, x, valid):
        '''
        gradients can flow back through 'gather_nd' and 'tensor_scatter_nd_update' so 
        they are differentiable. Its technically a sparse array but
         because we are using 10 jets max it doesn't really matter
        x: (B, N, Din)
        valid: (B, N) float: {0, 1}
        return (B, N, Dout)
        '''
        # get dims
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        Din = tf.shape(x)[2]
        # build a valid_bool from valid 
        valid_bool = tf.cast(valid > 0.5, tf.bool)  # (B, N)
        # gives indexes of valid tokens
        idx = tf.where(valid_bool)

        # extract only valid rows
        x_valid = tf.gather_nd(x, idx)  # (K,Din)
        # apply the arbitrary sublayer,
        # in this case its layer norm but could also be a small mlp
        y_valid = self.sublayer(x_valid)  # (K, Dout)
        Dout = tf.shape(y_valid)[-1]

        # create a zero array of right dims
        out = tf.zeros((B, N, Dout), dtype=y_valid.dtype)
        # scatter back
        out = tf.tensor_scatter_nd_update(out, idx, y_valid) 
        return out


