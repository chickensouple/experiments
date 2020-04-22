# Character Model RNNs

## Intro
Based off of the tensorflow tutorial here:
https://www.tensorflow.org/tutorials/text/text_generation
which in itself is based off of 
http://karpathy.github.io/2015/05/21/rnn-effectiveness/ .


This experiment tests generating character models with LSTM/GRU/Vanilla RNN on some simple text datasets.

## Basic Example Instructions
To train the shakespeare example given in the above links, run the following
in the directory this README.md is located in.

`python3 rnn_models.py --dataset tiny_shakespeare train --nepochs 10 --data_dir /tmp/rnn_model`

Then, to generate text from the trained model run

`python3 rnn_models.py --dataset tiny_shakespeare test --nchars 2000 --data_dir /tmp/rnn_model --temp 1.0`

This will generate 2000 (nchars) characters using a softmax temperature of 1.0.
Higher softmax temperatures will generate more random (and possibly less likely) output, while lower softmax temperatures will likely generate less random and more likely output.

To train on a custom blob of text, use `--dataset custom` and copy your text blob into the `data/custom_text.txt` file

## Dependencies
Code has been tested with the following packages:

* Python3.6
  * tensorflow 2.1
  * tensorflow-datasets 2.1
  * numpy 1.18.2
  * tqdm 4.31.1 
   