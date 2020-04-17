import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import csv

class CharProcessor(object):
    LOWER_CASE_CHARS = list("abcdefghijklmnopqrstuvwxyz")
    UPPER_CASE_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    NUMBERS = list("0123456789")
    PUNCTUATION = list(".,!?:;\'\"")
    WHITESPACE = list(" \n\t")
    SPECIAL = list("#$%&*\\/")
    ALL_DEFAULT = \
        LOWER_CASE_CHARS + \
        UPPER_CASE_CHARS + \
        NUMBERS + \
        PUNCTUATION + \
        WHITESPACE + \
        SPECIAL

    def __init__(self, 
        vocab,
        convert_to_lower=False,
        ignore_unknown=True):
        """
        Creates a CharProcessor Object capable of taking strings of texts
        and turning them into unique integers that represent each character.
        Each character defined in the vocab will be mapped to a unique integer.
        If a text to be converted has words not in the vocab, they can be ignored
        or mapped to a special <NAN> character.
        
        Arguments:
            vocab {list[string]} -- List of all characters in the vocab.
        
        Keyword Arguments:
            convert_to_lower {bool} -- If True, will treat both capital and lowercase letters
                the same (as lower case). (default: {False})
            ignore_unknown {bool} -- If True, will ignore characters not in the vocab
                rather than converting them to <NAN> (default: {True})
        """

        if convert_to_lower:
            # make sure vocab only contains lowercase
            vocab = sorted(set(map(lambda x: x.lower(), vocab)))

        self.token_to_idx = dict()
        self.idx_to_token = dict()
        for token in vocab:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

        # adding special characters for padding and N/A
        self.token_to_idx["<PAD>"] = len(self.token_to_idx)
        self.idx_to_token[len(self.idx_to_token)] = "<PAD>"
        self.token_to_idx["<NAN>"] = len(self.token_to_idx)
        self.idx_to_token[len(self.idx_to_token)] = "<NAN>"
        self.vocab_size = len(self.token_to_idx)

        self.convert_to_lower = convert_to_lower
        self.ignore_unknown = ignore_unknown

    def filter_vocab(self, char_list):
        filtered_char_list = []
        for token in char_list:
            if token in self.token_to_idx:
                filtered_char_list.append(token)
        return filtered_char_list

    def convert_to_int(self, char_list):
        """
        Converts a list of characters into their corresponding
        integers.
        
        Arguments:
            char_list {list[string]} -- List of tokens.
        
        Returns:
            list[int] -- List of mapped integers.
        """
        int_list = []
        for token in char_list:
            if self.convert_to_lower:
                cleaned_token = token.lower()
            else:
                cleaned_token = token

            if cleaned_token in self.token_to_idx:
                int_list.append(self.token_to_idx[cleaned_token])
            else:
                if not self.ignore_unknown:
                    int_list.append(self.token_to_idx["<NAN>"])
        return int_list

    def convert_to_char(self, int_list):
        """
        Converts a list of integers to their corresponding characters.
        
        Arguments:
            int_list {list[int]}
        
        Returns:
            list[string]
        """
        char_list = []
        for idx in int_list:
            if idx < self.vocab_size:
                char_list.append(self.idx_to_token[idx])
            else:
                raise Exception("Invalid idx")
        return char_list

def load_data(dataset_name, get_val=False):
    """
    Loads datasets as one big string.
    
    Arguments:
        dataset_name {string} -- Name of dataset.
            should be in the list ["tiny_shakespeare", "imdb_reviews"]
        get_val {bool} -- If True, get validation data instead of training data.
    
    Returns:
        string -- the dataset.
    """
    if dataset_name == "tiny_shakespeare":
        if get_val:
            data = tfds.load(name='tiny_shakespeare')['validation']
        else:
            data = tfds.load(name='tiny_shakespeare')['train']
        text = [x for x in data][0]['text'].numpy().decode('utf-8')
        return text
    elif dataset_name == "imdb_reviews":
        if get_val:
            data = tfds.load(name='imdb_reviews')['test']
        else:
            data = tfds.load(name='imdb_reviews')['train']
        text = [x['text'].numpy().decode('utf-8') for x in data]
        text = "".join(text)
        return text
    elif dataset_name == "trump_tweets":
        if get_val:
            return None
        with open("data/trump_tweets.csv") as csvfile:
            data = csv.reader(csvfile, delimiter=",")
            text = []
            for idx, row in enumerate(data):
                if idx == 0:
                    continue
                text.append(row[1])
                text.append("\n")
            text = "".join(text)

            text_processor = CharProcessor(CharProcessor.ALL_DEFAULT)
            text = text_processor.filter_vocab(text)
            text = "".join(text)
        return text
    elif dataset_name == "custom":
        if get_val:
            return None
        with open("data/custom_text.txt") as file:
            text = file.read()
            text_processor = CharProcessor(CharProcessor.ALL_DEFAULT)
            text = text_processor.filter_vocab(text)
            text = "".join(text)
            return text
    else:
        raise Exception("Not a valid dataset_name.")

def create_char_pred_ds(int_list, seq_len=100, batch_size=32):
    """
    Creates a tensorflow dataset for character prediction from a long string.
    This will split the text into sequences of length <seq_len>
    (dropping extra characters that remain after the division),
    and then will batch sequences into batches of <batch_size>.

    Arguments:
        int_list {list[int]} -- List of ints, where each int represents a character.
            This can come from using the CharProcessor() on a dataset loaded by load_data()
    
    Keyword Arguments:
        seq_len {int} -- (default: {100})
        batch_size {int} -- (default: {32})
    
    Returns:
        [tf.data.Dataset] -- tensorflow dataset
    """
    curr_char = tf.convert_to_tensor(int_list[:-1])
    next_char = tf.convert_to_tensor(int_list[1:])
    
    ds = tf.data.Dataset.from_tensor_slices((curr_char, next_char))
    ds = ds.batch(seq_len, drop_remainder=True)
    ds = ds.batch(batch_size)
    return ds


if __name__ == "__main__":
    # Run this file to simply load
    # a dataset, convert it to integers, 
    # and create a tensorflow dataset.
    text = load_data("custom", get_val=False)
    vocab = sorted(set(text))

    print(text)
    text_processor = CharProcessor(vocab)
    int_list = text_processor.convert_to_int(text)
    ds = create_char_pred_ds(int_list)
