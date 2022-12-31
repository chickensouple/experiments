import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import pickle
import plotly.tools as tls
import plotly.offline

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def get_mnist_dataset():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    
    
    ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test

class SimpleNetwork(tf.keras.Model):
    def __init__(self, batch_norm=False):
        super().__init__()
        self.batch_norm = True
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.hidden_layers = [
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu')]

        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization() for _ in self.hidden_layers
        ]
        self.output_layer = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.flatten(inputs)
        hidden_outputs = []
        for batch_norm_layer, layer in zip(self.batch_norm_layers, self.hidden_layers):
            x = layer(x)
            if self.batch_norm:
                x = batch_norm_layer(x)
            hidden_outputs.append(x)
        x = self.output_layer(x)
        return x, hidden_outputs


def get_model_metrics(model, dataset):
    data_dict = dict()
    for x, y in dataset:
        y_pred, layer_outputs = model(x)
        for idx, layer_output in enumerate(layer_outputs):
            if idx not in data_dict.keys():
                data_dict[idx] = []
            data_dict[idx].extend(tf.norm(layer_output, axis=-1))
    return data_dict




def generate_plots(epoch_data_dict_list, plot_epoch_idx):
    max_norms = dict()    
    for epoch_idx, epoch_data_dict in enumerate(epoch_data_dict_list):
        for key, val in epoch_data_dict.items():
            if key not in max_norms:
                max_norms[key] = 0
            max_norms[key] = max(max_norms[key], np.max(val))


    cmap = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(0, len(epoch_data_dict_list)), plt.get_cmap("summer"))
    for epoch_idx, epoch_data_dict in enumerate(epoch_data_dict_list):
        # import pdb; pdb.set_trace()
        color = cmap.to_rgba(epoch_idx)
        for key, val in epoch_data_dict.items():
            plt.figure(key)
            
            if epoch_idx == 0:
                plt.title("Layer {} outputs".format(key))

            bin_count, bin_edges = np.histogram(val, bins=10, range=[0, max_norms[key]], density=True)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            bin_widths = bin_edges[1] - bin_edges[0]

            mean_norm = np.mean(val)
            std_norm = np.std(val)
            fitted_normal = scipy.stats.norm(loc=mean_norm, scale=std_norm)
            normal_x = np.linspace(0, max_norms[key], 100)
            normal_y = fitted_normal.pdf(normal_x)
            plt.plot(normal_x, normal_y, c=color)

            if epoch_idx == plot_epoch_idx:
                plt.bar(bin_centers, bin_count, width=bin_widths*0.8, color=color, alpha=0.5)



if __name__ == "__main__":
    # (ds_train, ds_test) = get_mnist_dataset()
    # model = SimpleNetwork(True)

    # num_epochs = 5
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # epoch_data_dict_list = []
    # for epoch in range(num_epochs):
    #     epoch_loss_avg_train = tf.keras.metrics.Mean()
    #     epoch_accuracy_train = tf.keras.metrics.SparseCategoricalAccuracy()
    #     epoch_loss_avg_test = tf.keras.metrics.Mean()
    #     epoch_accuracy_test = tf.keras.metrics.SparseCategoricalAccuracy()
    #     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
    #     for x, y in ds_train:
    #         with tf.GradientTape() as tape:
    #             y_pred, _ = model(x)
    #             loss_value = loss(y, y_pred)
                
    #         grads = tape.gradient(loss_value, model.trainable_variables)

    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #         # Track progress
    #         epoch_loss_avg_train.update_state(loss_value)
    #         epoch_accuracy_train.update_state(y, y_pred)

    #     epoch_data_dict = get_model_metrics(model, ds_train)
    #     epoch_data_dict_list.append(epoch_data_dict)

    #     print("Epoch {:03d}: Train Loss: {:.3f}, Train Accuracy: {:.3%}".format(epoch,
    #                                                                 epoch_loss_avg_train.result(),
    #                                                                 epoch_accuracy_train.result()))
    # pickle.dump(epoch_data_dict_list, open("data2.pickle", "wb"))



    epoch_data_dict_list = pickle.load(open("data2.pickle", "rb"))
    
    generate_plots(epoch_data_dict_list, 0)

    plt.show()


