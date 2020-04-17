import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import argparse
import os

from nlp_dataset import *

# gpu setup copied from https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class SimpleCharacterRNN(tf.keras.Model):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim=256, 
        rnn_units=512,
        rnn_type="gru"):
        super().__init__()

        self.vocab_size = vocab_size
        self.encoding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        if rnn_type == "lstm":
            self.lstm = tf.keras.layers.LSTM(
                units=rnn_units,
                return_sequences=True,
                return_state=True)
        elif rnn_type == "gru":
            self.lstm = tf.keras.layers.GRU(
                units=rnn_units,
                return_sequences=True,
                return_state=True)
        else:
            raise Exception("Not a valid rnn_type.")
        self.decoding = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, initial_state=None):
        """
        Returns both the output and the state of the RNN
        """
        embedding = self.encoding(inputs)
        ret = self.lstm(embedding, initial_state=initial_state)
        output = ret[0]
        state = ret[1:]
        decoding = self.decoding(output)
        return decoding, state

    def sample(self, starting_char, num_chars, temperature=1.0):

        curr_char = tf.reshape(starting_char, (1, 1))
        curr_state = None

        # (hidden state, cell state)
        char_list = [starting_char]
        for _ in range(num_chars):
            output, curr_state = self.call(curr_char, initial_state=curr_state)
            char_logits = np.reshape(output, (-1))
            char_logits = char_logits / temperature

            dist = tfp.distributions.Categorical(logits=char_logits, dtype=tf.int32)
            char_token = tfp.distributions.Sample(dist, sample_shape=(1,)).sample()
            curr_char = tf.reshape(char_token, (1, 1))
            char_list.append(int(char_token))
        return char_list

def character_prediction_loss(y, y_hat):
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=y_hat), 
        axis=1)
    return loss

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

def test(model, char_processor, num_chars, starting_char, temp):
    """
    Tests a model
    
    Arguments:
        model {SimpleCharacterRNN} -- Model to draw sampled test from.
        char_processor {CharProcessor} -- Character Processor to convert
            from model output into text.
        num_chars {int} -- Number of characters to sample.
        starting_char {string} -- Starting character to give model.
        temp {float} -- Temperature for softmax    
    """
    idx_list = model.sample(
        starting_char=char_processor.convert_to_int(starting_char)[0], 
        num_chars=num_chars,
        temperature=temp)
    sampled_text = char_processor.convert_to_char(idx_list)
    sampled_text = ''.join(sampled_text)
    print("Predicted Text: ")
    print(sampled_text)
    return sampled_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Test Character LSTM models")
    parser.add_argument(
        "--dataset",
        action="store",
        type=str,
        choices=["tiny_shakespeare", "imdb_reviews", "trump_tweets", "custom"],
        default="tiny_shakespeare",
        help="Name of dataset to use. \
              Testing must use same as training, as it is used to \
              generate the vocabulary.")
    parser.add_argument(
        "--rnn_type",
        action="store",
        type=str,
        choices=["gru", "lstm", "rnn"],
        default="gru",
        help="Which kind of rnn to use. GRU, LSTM or a vanilla RNN.")
    
    subparser = parser.add_subparsers(
        help="Train or Test the character model. \
              Choose an option and add -h for further options.",
        dest="cmd")
    train_subparser = subparser.add_parser("train",
        help="Train a model.")
    train_subparser.add_argument(
        "--nepochs",
        action="store", 
        type=int,
        default=10,
        help="Number of epochs to train for.")
    train_subparser.add_argument(
        "--data_dir",
        action="store",
        type=str,
        default="/tmp/rnn_char_model",
        help="Directory to save model and log data in.")

    test_subparser = subparser.add_parser("test", 
        help="Draw sampled text from a model.")
    test_subparser.add_argument(
        "--data_dir",
        action="store",
        type=str,
        default="/tmp/rnn_char_model",
        help="Directory to load model data from.\
              Should be the same as the argument given to train.")
    test_subparser.add_argument(
        "--nchars", 
        type=int,
        default=1000,
        help="Number of characters to predict")
    test_subparser.add_argument(
        "--starting_char", 
        type=str,
        default=".",
        help="Number of characters to predict")
    test_subparser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperature for softmax function during prediction.\
              Takes on values (0, inf) exclusive.\
              Higher temperature results in more random predictions.")
    args = parser.parse_args()

    text = load_data(args.dataset)
    vocab = sorted(set(text))
    char_processor = CharProcessor(vocab)
    vocab_size = len(vocab)

    model = SimpleCharacterRNN(
        vocab_size=vocab_size,
        rnn_type=args.rnn_type)

    if args.cmd == "train":
        model_dir = os.path.join(args.data_dir, "models")
        log_dir = os.path.join(args.data_dir, "logs")
        val_text = load_data(args.dataset, get_val=True)
        train(model, text, val_text, model_dir, log_dir, args.nepochs)
    elif args.cmd == "test":
        model_path = os.path.join(args.data_dir, "models", "model_final.ckpt")
        model.load_weights(model_path)
        test(model, char_processor, args.nchars, args.starting_char, args.temp)
    else:
        raise Exception("Not a valid cmd.")
     
