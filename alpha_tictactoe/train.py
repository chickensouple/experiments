import numpy as np
import tensorflow as tf
from tqdm import tqdm

from minmax import OptimalActor
from mcts import MCTSActor
from tictactoe import TicTacToe
from game_base import GameBase, run_game
from neural_nets import ValueNetwork
from replay_buffer import ReplayBuffer

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

if __name__ == "__main__":
    value_network = ValueNetwork()
    ttt_mcts = TicTacToe()
    mcts_actor = MCTSActor(ttt_mcts, value_network=value_network, num_expansions=300)

    optimal_data = np.load("data/minmax_cache.npy")
    optimal_actor = OptimalActor(optimal_data)

    ttt = TicTacToe()

    optimizer = tf.keras.optimizers.Adam()
    buffer_config = {"states": (3, 3, 1), "values": 1}
    replay_buffer = ReplayBuffer(100, buffer_config)
    for i in range(100):
        if i % 2 == 0:
            data_dict = run_game(ttt, optimal_actor, mcts_actor)
        else:
            data_dict = run_game(ttt, mcts_actor, optimal_actor)

        result = data_dict["game_status"]
        if result == GameBase.Status.PLAYER1_WIN:
            value = 1
        elif result == GameBase.Status.PLAYER2_WIN:
            value = -1
        else:
            value = 0

        states = np.array([x[:9].reshape(3, 3, 1) for x in data_dict["states"]])
        values = np.array([[value] for _ in data_dict["states"]])
        data_batch = {"states": states, "values": values}
        replay_buffer.add_data(data_batch)

        if i > 5:
            batch_size = 32
            sampled_data = replay_buffer.sample(batch_size)

            x = np.array(sampled_data["states"], dtype=np.float32)
            y = np.array(sampled_data["values"], dtype=np.float32)

            with tf.GradientTape() as tape:
                values = value_network(x)
                loss = tf.reduce_mean(tf.square(values - y))
            grads = tape.gradient(loss, value_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, value_network.trainable_variables))
            print("{} -- Loss: {}".format(i, loss))

    value_network.save_weights("models/model.ckpt")


