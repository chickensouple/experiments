import tensorflow as tf
import numpy as np


def multihead_projection(input, proj_weights):
    '''
    input: (..., n_in)
    proj_weights: (n_heads, n_in, n_out)

    out: (..., n_heads, n_out)
    '''
    return tf.einsum("...i,hio->...ho", input, proj_weights)


def multihead_scaled_dot_attention(query, key, value, mask):
    '''
    |mask| has true if valid and false if not valid. |mask[i][j] == true| means that the i'th output value
    can attend to the j'th key/value. The mask is shared among all heads.

    query: (..., n_seq, n_heads, n_qk_proj)
    key: (..., n_seq, n_heads, n_qk_proj)
    value: (..., n_seq, n_heads, n_val_proj)
    mask: (..., n_seq, n_seq)

    out: (..., n_seq, n_heads, n_val_proj)
    '''
    n_seq = tf.shape(key)[-3]
    sqrt_n_seq = np.sqrt(n_seq)

    n_heads = tf.shape(key)[-2]

    # |query_key_dot_logits| is of shape (..., n_heads, n_seq, n_seq)
    query_key_dot_logits = tf.einsum(
        "...ihj,...khj->...hik", query, key) / sqrt_n_seq

    # Create |stacked_neg_inf_mask| of shape (..., n_heads, n_seq, n_seq) where
    # |stacked_neg_inf_mask[..., :, i, j] == 0 if mask[..., i, j] == True and -inf otherwise.
    zeros = np.zeros(mask.shape, dtype=np.float32)
    neg_inf_mask = tf.where(mask, zeros, -np.inf)
    stacked_neg_inf_mask = tf.stack(
        [neg_inf_mask for _ in range(n_heads)], axis=-3)

    # Add |stacked_neg_inf_mask| so that any elements that are -inf will drive logits to -inf.
    query_key_dot_logits_masked = query_key_dot_logits + stacked_neg_inf_mask

    # |prob_weights| is of shape (..., n_heads, n_seq, n_seq)
    prob_weights = tf.nn.softmax(query_key_dot_logits_masked, axis=-1)
    return tf.einsum("...hij,...jhk->...ihk", prob_weights, value)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads, n_q, n_k, n_v, n_qk_proj, n_v_proj, n_out):
        super().__init__()
        self.n_heads = n_heads
        self.n_q = n_q
        self.n_k = n_k
        self.n_v = n_v
        self.n_qk_proj = n_qk_proj
        self.n_v_proj = n_v_proj
        self.n_out = n_out

        self.query_proj_weight = self.add_weight(
            shape=(self.n_heads, n_q, n_qk_proj), initializer="glorot_normal", trainable=True
        )
        self.key_proj_weight = self.add_weight(
            shape=(self.n_heads, n_k, n_qk_proj), initializer="glorot_normal", trainable=True
        )
        self.value_proj_weight = self.add_weight(
            shape=(self.n_heads, n_v, n_v_proj), initializer="glorot_normal", trainable=True
        )
        self.out_proj_weight = self.add_weight(
            shape=(self.n_heads * n_v_proj, n_out), initializer="glorot_normal", trainable=True
        )

    def _check_inputs(self, query, key, value, mask):
        # Check that the number of dimensions is the same and at least 2.
        assert((len(query.shape) == len(key.shape))
               and (len(key.shape) == len(value.shape))
               and (len(value.shape) == len(mask.shape)))
        assert(len(query.shape) >= 2)

        # Check that the sequence length is the same.
        n_seq = query.shape[-2]
        assert((n_seq == key.shape[-2]) and (n_seq == value.shape[-2]))
        assert((n_seq == mask.shape[-1]) and (n_seq == mask.shape[-2]))

        # Check that the batch dimensions are all the same.
        assert((query.shape[:-2] == key.shape[:-2])
               and (key.shape[:-2] == value.shape[:-2])
               and (value.shape[:-2] == mask.shape[:-2]))

        # Check that query, key, values have the right embedding sizes.
        assert(query.shape[-1] == self.n_q)
        assert(key.shape[-1] == self.n_k)
        assert(value.shape[-1] == self.n_v)

    def call(self, query, key, value, mask):
        '''
        query: (..., n_seq, n_q)
        key: (..., n_seq, n_k)
        value: (..., n_seq, n_v)
        mask: (..., n_seq, n_seq)
        '''
        self._check_inputs(query, key, value, mask)
        n_seq = query.shape[-2]

        query_proj = multihead_projection(query, self.query_proj_weight)
        key_proj = multihead_projection(key, self.key_proj_weight)
        value_proj = multihead_projection(value, self.value_proj_weight)

        # (..., n_seq, n_heads, n_v_proj)
        out = multihead_scaled_dot_attention(
            query_proj, key_proj, value_proj, mask)

        # Combine all the heads together so that |out_reshaped| is (..., n_seq, n_heads * n_v_proj)
        flattened_out_shape = out.shape[:-3] + \
            [n_seq, self.n_heads * self.n_v_proj]
        out_reshaped = tf.reshape(out, flattened_out_shape)
        return tf.linalg.matmul(out_reshaped, self.out_proj_weight)


if __name__ == "__main__":
    B = 2
    N = 3
    H = 4
    n_q = 14
    n_k = 15
    n_v = 16
    n_qk_proj = 5
    n_v_proj = 6
    np.random.seed(0)

    query = np.ones((B, N, n_q), dtype=np.float32)
    key = np.ones((B, N, n_k), dtype=np.float32)
    val = np.random.random((B, N, n_v)).astype(np.float32)
    query_proj_weights = np.ones((H, n_q, n_qk_proj), dtype=np.float32)
    key_proj_weights = np.ones((H, n_k, n_qk_proj), dtype=np.float32)
    val_proj_weights = np.ones((H, n_v, n_v_proj), dtype=np.float32)

    print("shape1: {}".format(query.shape))
    print("shape2: {}".format(query_proj_weights.shape))
    query_proj = multihead_projection(query, query_proj_weights)
    key_proj = multihead_projection(key, key_proj_weights)
    val_proj = multihead_projection(val, val_proj_weights)

    mask = np.full((B, N, N), True)

    out = multihead_scaled_dot_attention(query_proj, key_proj, val_proj, mask)
    print(out)

    mask2 = np.full((B, N, N), True)
    mask2[:, 0, 0] = False
    out2 = multihead_scaled_dot_attention(
        query_proj, key_proj, val_proj, mask2)
    print(out2)

    n_out = 11
    multihead_attention = MultiHeadAttention(
        H, n_q, n_k, n_v, n_qk_proj, n_v_proj, n_out)
    out3 = multihead_attention(query, key, val, mask2)
    print(out3)
    import pdb
    pdb.set_trace()
    pass
