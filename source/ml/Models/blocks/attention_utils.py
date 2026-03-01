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

def _delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return tf.atan2(tf.sin(dphi), tf.cos(dphi))

def build_pairwise_jet_features(
    jets,
    valid,
    pt_idx=0,
    eta_idx=1,
    phi_idx=2,
    m_idx=3,
    btag_idx=4,
    mode="full",
):
    """
    Build pairwise jet features for relation-aware attention.
    This is only used in v5x3 and is not used in the thesis, but is in the codebase for future work.
    parts of this code function were modified by GenAI
    jets: (B, N, F)
    valid: (B, N) in {0,1}
    Returns: (B, N, N, F_pair), pair_mask (B, N, N)
    """
    pt = jets[..., pt_idx]
    eta = jets[..., eta_idx]
    phi = jets[..., phi_idx]
    m = jets[..., m_idx]

    # mask invalid jets to zeros
    v = tf.cast(valid, jets.dtype)
    pt = pt * v
    eta = eta * v
    phi = phi * v
    m = m * v

    # Expand dims for pairwise ops
    pt_i = pt[:, :, None]
    pt_j = pt[:, None, :]
    eta_i = eta[:, :, None]
    eta_j = eta[:, None, :]
    phi_i = phi[:, :, None]
    phi_j = phi[:, None, :]
    m_i = m[:, :, None]
    m_j = m[:, None, :]

    d_eta = eta_i - eta_j
    d_phi = _delta_phi(phi_i, phi_j)
    d_r = tf.sqrt(tf.maximum(d_eta * d_eta + d_phi * d_phi, 0.0))

    if mode in ("full", "mid"):
        pt_ratio = pt_i / (pt_j + tf.keras.backend.epsilon())

    if mode == "full":
        # invariant mass for pair (i,j)
        # make 4-vectors from (pt, eta, phi, m)
        px_i = pt_i * tf.cos(phi_i)
        py_i = pt_i * tf.sin(phi_i)
        pz_i = pt_i * tf.sinh(eta_i)
        e_i = tf.sqrt(tf.maximum(m_i * m_i + px_i * px_i + py_i * py_i + pz_i * pz_i, 0.0))

        px_j = pt_j * tf.cos(phi_j)
        py_j = pt_j * tf.sin(phi_j)
        pz_j = pt_j * tf.sinh(eta_j)
        e_j = tf.sqrt(tf.maximum(m_j * m_j + px_j * px_j + py_j * py_j + pz_j * pz_j, 0.0))

        e_ij = e_i + e_j
        px_ij = px_i + px_j
        py_ij = py_i + py_j
        pz_ij = pz_i + pz_j
        m2_ij = e_ij * e_ij - (px_ij * px_ij + py_ij * py_ij + pz_ij * pz_ij)
        m_ij = tf.sqrt(tf.maximum(m2_ij, 0.0))

    if btag_idx is not None:
        btag = jets[..., btag_idx] * v
        btag_i = btag[:, :, None]
        btag_j = btag[:, None, :]
        btag_prod = btag_i * btag_j
    else:
        btag_prod = tf.zeros_like(d_r)

    pair_mask = v[:, :, None] * v[:, None, :]

    if mode == "full":
        pair_feats = tf.stack([d_r, d_eta, d_phi, m_ij, pt_ratio, btag_prod], axis=-1)
    elif mode == "mid":
        pair_feats = tf.stack([d_r, d_eta, d_phi, pt_ratio, btag_prod], axis=-1)
    elif mode == "lite":
        pair_feats = tf.stack([d_r, d_eta, d_phi, btag_prod], axis=-1)
    else:
        raise ValueError(f"Unknown pairwise feature mode: {mode}")
    pair_feats = pair_feats * pair_mask[..., None]

    return pair_feats, pair_mask

def build_jet_mask(jets, mask_value):
    ''' This build a per-jet 'is valid' mask by checking if any
     feature in a jet row is not equal to the padding value. 
     If there is a non padding value it is classed as valid'''
    # jets (batch_size, num_jets, jet_features)
    # mask_value (scalar)
    is_valid = tf.reduce_any(tf.not_equal(jets, mask_value), axis=-1)  # (batch_size, num_jets)
    return tf.cast(is_valid, tf.float32)

def apply_pad_zeros(x, valid):
    ''' this is going to zero out padded
     rows to make sure that any operation that sees it gets exactly 0.0
     keras doesnt like -inf so we have to go with zero
    x: (B, N, D), valid: (B, N)
    '''
    return x*tf.expand_dims(valid, axis=-1)

def make_attn_mask(valid):
    '''we need to build an attention mask that is compatible with the keras MHA layer
    so:
    True = keep, false = mask
    valid: (B, N)

    for self attention the query length and
     the key length(num jets) are the same so keras can broadcast
    '''
    keep = tf.cast(valid > 0.5, tf.bool)  # (B, N)
    # in self attention the query length is N (same as key length), so we need to
    # broadcast the mask to (B, N, N)
    return tf.expand_dims(keep, axis=1)
