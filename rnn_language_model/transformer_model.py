import tensorflow as tf
import numpy as np
import math
from keras.utils.layer_utils import count_params
from nlp_dataset import *
import os
from tqdm import tqdm


def multihead_projection(inp, proj_weights):
    '''
    inp: (..., n_in)
    proj_weights: (n_heads, n_in, n_out)

    out: (..., n_heads, n_out)
    '''
    return tf.einsum("...i,hio->...ho", inp, proj_weights)


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

    def _check_inps(self, query, key, value, mask):
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

        out: (..., n_seq, n_out)
        '''
        self._check_inps(query, key, value, mask)
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


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    K_DEFAULT_MAX_POS = 100

    def __init__(self, n_embed, K=10000):
        super().__init__()
        self.n_embed = n_embed
        self.K = K
        self.len1 = int(math.ceil(self.n_embed / 2.0))
        self.len2 = self.n_embed - self.len1

        # [n_seq, n_embed]
        self.pos_embedding = self._compute_encoding(
            SinusoidalPositionalEncoding.K_DEFAULT_MAX_POS)

    def _compute_encoding(self, max_pos):
        pos1 = np.stack([np.arange(0, max_pos) for _ in range(self.len1)]).T
        inner_exp1 = np.stack([np.arange(0, self.len1)
                              for _ in range(max_pos)])

        pos2 = np.stack([np.arange(0, max_pos) for _ in range(self.len2)]).T
        inner_exp2 = np.stack([np.arange(0, self.len2)
                               for _ in range(max_pos)])

        even_rows = np.sin(
            pos1 / np.power(self.K, (2.0 / self.n_embed) * inner_exp1))
        odd_rows = np.cos(
            pos2 / np.power(self.K, (2.0 / self.n_embed) * inner_exp2))

        out = np.empty((max_pos, self.n_embed), dtype=np.float32)
        out[:, ::2] = even_rows
        out[:, 1::2] = odd_rows
        return out

    def call(self, inp):
        '''
        inp: (..., n_seq, n_embed)
        '''

        n_seq = inp.shape[-2]
        # Update pos embedding if sequence length is too short.
        if (n_seq > self.pos_embedding.shape[0]):
            self.pos_embedding = self._compute_encoding(n_seq)

        return inp + self.pos_embedding[0:n_seq, :]


class SimpleSelfAttentionLayer(tf.keras.Model):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.n_proj = int(self.n_embed / self.n_heads)
        assert(self.n_heads * self.n_proj == self.n_embed)

        self.multihead_attention = MultiHeadAttention(
            self.n_heads, self.n_embed, self.n_embed, self.n_embed, self.n_proj, self.n_proj, self.n_embed)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        self.dense1 = tf.keras.layers.Dense(
            units=self.n_embed, activation='relu')
        self.dense2 = tf.keras.layers.Dense(
            units=self.n_embed, activation=None)

    def _create_mask(self, key_shape, n_seq):
        mask = np.full((n_seq, n_seq), False)
        mask[np.tril_indices(n=n_seq)] = True
        stacked_mask = np.full(
            key_shape[:-2] + [n_seq, n_seq], False) + mask

        return stacked_mask

    def call(self, inp):
        '''
        inp is (..., n_seq, n_embed)
        '''
        n_seq = inp.shape[-2]
        x0 = inp

        # (..., n_seq, n_embed)
        mask = self._create_mask(inp.shape, n_seq)
        x1 = self.multihead_attention(x0, x0, x0, mask)

        # (..., n_seq, n_dim)
        x2 = self.layer_norm1(x1 + x0)

        # (..., n_seq, n_dim)
        x3 = self.dense2(self.dense1(x2))

        # (..., n_seq, n_dim)
        x4 = self.layer_norm2(x3 + x2)

        return x4


class SimpleAttentionLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, n_embed, n_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_heads = n_heads

        self.vocab_embedding = tf.keras.layers.Embedding(
            vocab_size, n_embed)

        self.pos_encoding = SinusoidalPositionalEncoding(
            n_embed=self.n_embed)

        self.self_attention1 = SimpleSelfAttentionLayer(
            self.n_embed, self.n_heads)
        self.self_attention2 = SimpleSelfAttentionLayer(
            self.n_embed, self.n_heads)

    def call(self, inp):
        '''
        inp is (..., n_seq)
        '''

        x0 = inp
        # (..., n_seq, n_embed)
        x1 = self.vocab_embedding(inp)

        # (..., n_seq, n_embed)
        x2 = self.pos_encoding(x1)

        # (..., n_seq, n_embed)
        x3 = self.self_attention1(x2)

        # (..., n_seq, n_embed)
        x4 = self.self_attention2(x3)

        return x4


def train_step(x, y, model, optimizer):
    with tf.GradientTape() as tape:
        y_hat, _ = model(x)
        loss = character_prediction_loss(y, y_hat)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train(model, train_text_data, val_text_data, modeldir, logdir, epochs):
    """
    Trains a model.

    Arguments:
        model {SimpleCharacterRNN} -- Model to train.
        train_text_data {string} -- Training data.
        val_text_data {string} -- Validation data.
        modeldir {string} -- Directory to save model weights in.
        logdir {string} -- Directory to save logs in.
        epochs {int} -- Number of epochs to train for.
    """
    train_logdir = os.path.join(logdir, "train")
    val_logdir = os.path.join(logdir, "val")
    train_summary_writer = tf.summary.create_file_writer(train_logdir)
    val_summary_writer = tf.summary.create_file_writer(val_logdir)

    vocab = sorted(set(train_text_data))
    char_processor = CharProcessor(vocab)
    train_idx_list = char_processor.convert_to_int(train_text_data)
    train_ds = create_char_pred_ds(train_idx_list)

    if (val_text_data is not None):
        val_idx_list = char_processor.convert_to_int(val_text_data)
        val_ds = create_char_pred_ds(val_idx_list)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")

    for i in range(epochs):
        train_loss.reset_states()
        val_loss.reset_states()

        train_ds = train_ds.shuffle(10000)
        for (x, y) in tqdm(train_ds):
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            import pdb
            pdb.set_trace()
            loss = train_step(x, y, model, optimizer)

        # Evaluate loss functions after every epoch of training
        for (x, y) in train_ds:
            y_hat, _ = model(x)
            loss = character_prediction_loss(y, y_hat)
            train_loss(loss)

        if val_text_data is not None:
            for (x, y) in tqdm(val_ds):
                y_hat, _ = model(x)
                loss = character_prediction_loss(y, y_hat)
                val_loss(loss)

        print("Epoch {}: train loss = {}, val_loss = {}".
              format(i, train_loss.result(), val_loss.result()))
        model_path = os.path.join(modeldir, "model_" + str(i) + ".ckpt")
        model.save_weights(model_path)
        with train_summary_writer.as_default():
            tf.summary.scalar("train_loss", train_loss.result(), i)
        with val_summary_writer.as_default():
            tf.summary.scalar("val_loss", val_loss.result(), i)

    model_path = os.path.join(modeldir, "model_final.ckpt")
    model.save_weights(model_path)


if __name__ == "__main__":
    # B = 2
    # N = 3
    # H = 4
    # n_q = 14
    # n_k = 15
    # n_v = 16
    # n_qk_proj = 5
    # n_v_proj = 6
    # np.random.seed(0)

    # query = np.ones((B, N, n_q), dtype=np.float32)
    # key = np.ones((B, N, n_k), dtype=np.float32)
    # val = np.random.random((B, N, n_v)).astype(np.float32)
    # query_proj_weights = np.ones((H, n_q, n_qk_proj), dtype=np.float32)
    # key_proj_weights = np.ones((H, n_k, n_qk_proj), dtype=np.float32)
    # val_proj_weights = np.ones((H, n_v, n_v_proj), dtype=np.float32)

    # print("shape1: {}".format(query.shape))
    # print("shape2: {}".format(query_proj_weights.shape))
    # query_proj = multihead_projection(query, query_proj_weights)
    # key_proj = multihead_projection(key, key_proj_weights)
    # val_proj = multihead_projection(val, val_proj_weights)

    # mask = np.full((B, N, N), True)

    # out = multihead_scaled_dot_attention(query_proj, key_proj, val_proj, mask)
    # print(out)

    # mask2 = np.full((B, N, N), True)
    # mask2[:, 0, 0] = False
    # out2 = multihead_scaled_dot_attention(
    #     query_proj, key_proj, val_proj, mask2)
    # print(out2)

    # n_out = 11
    # multihead_attention = MultiHeadAttention(
    #     H, n_q, n_k, n_v, n_qk_proj, n_v_proj, n_out)
    # out3 = multihead_attention(query, key, val, mask2)
    # print(out3)
    # import pdb
    # pdb.set_trace()
    # pass

    text = load_data("tiny_shakespeare")
    vocab = sorted(set(text))
    char_processor = CharProcessor(vocab)
    vocab_size = len(vocab)

    model = SimpleAttentionLanguageModel(
        vocab_size=vocab_size, n_embed=512, n_heads=8)
    num_params = count_params(model.trainable_weights)

    train(model, text, None, "", "", 100)
