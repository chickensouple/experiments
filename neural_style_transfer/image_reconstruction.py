import numpy as np
import tensorflow as tf
import imageio
import cv2
import matplotlib.pyplot as plt
from vgg_utils import *


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

def style_loss(y_feats, y_hat_feats):
    assert(len(y_feats) == len(y_hat_feats))
    loss = 0

    def gram_matrix(feat):
        # This feat matrix is transposed from the paper definition
        f_l = tf.reshape(feat, (-1, feat.shape[-1]))

        M_l = f_l.shape[0]
        N_l = f_l.shape[1]

        gram = tf.linalg.matmul(tf.transpose(f_l), f_l) / (2 * M_l * N_l)

        return gram

    for (feat1, feat2) in zip(y_feats, y_hat_feats):
        assert(feat1.shape == feat2.shape)
        
        G_l = gram_matrix(feat1)
        A_l = gram_matrix(feat2)
        loss += tf.reduce_sum(tf.square(A_l - G_l))
    return loss

def content_loss(y_feats, y_hat_feats):
    assert(len(y_feats) == len(y_hat_feats))
    loss = 0
    for feat1, feat2 in zip(y_feats, y_hat_feats):
        loss += tf.reduce_sum(tf.square(feat1 - feat2))
    return 0.5 * loss

def reconstruct(target_features, initial_guess, loss_func, niter):
    gen_image = tf.Variable(initial_guess)
    variables = [gen_image]

    opt = tf.keras.optimizers.Adam(learning_rate=10.)
    for i in range(niter):
        with tf.GradientTape() as tape:
            gen_features = layer1_model(gen_image)
            loss = loss_func(target_features, gen_features)
        grads = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grads, variables))
        print("Iteration {}: {}".format(i, loss))

    output = np.array(gen_image[0, ...])
    output = preprocess_image_inv(output)
    return output

if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser(description="Reconstructing Images from only feature maps.")
    parser.add_argument(
        "--image",
        action="store",
        type=str,
        required=True,
        help="Path to image to reconstruct.")
    parser.add_argument(
        "--layers",
        nargs="+",
        help="Name of vgg layers to use features of.\
              Choose from the following: \
              [block1_conv1, block1_conv2, block1_pool, \
               block2_conv1, block2_conv2, block2_pool, \
               block3_conv1, block3_conv2, block3_conv3, \
               block3_conv4, block3_pool, block4_conv1, \
               block4_conv2, block4_conv3, block4_conv4, \
               block4_pool, block5_conv1, block5_conv2, \
               block5_conv3, block5_conv4, block5_pool]")
    parser.add_argument(
        "--niter",
        action="store",
        type=int,
        default=100,
        help="Number of iterations to train the reconstruction for"
    )
    parser.add_argument(
        "--type",
        action="store",
        type=str,
        choices=["content", "style"],
        default="content",
        help="Choose between reconstructing content and reconstructing style"
    )
    args = parser.parse_args()
    
    target_image = imageio.imread(args.image)

    # If image is greyscale, copy to all color coordinates
    if len(target_image.shape) == 2:
        target_image = np.stack([target_image for _ in range(3)])

    # If image has an alpha channel, get rid of it
    if target_image.shape[2] == 4:
        target_image = target_image[:, :, :3]

    vgg_layers = VGGLayers(target_image.shape)
    vgg_layer_names = vgg_layers.get_layer_names()
    for layer in args.layers:
        if layer not in vgg_layer_names:
            raise Exception("{} is not a valid layer. Only the following are valid: {}".format(layer, vgg_layer_names))
    layer1_model = vgg_layers.get_model(args.layers)

    # getting target features
    target_image_preprocessed = np.array([preprocess_image(target_image)])
    target_features = layer1_model(target_image_preprocessed)

    gen_image = np.random.randn(*target_image_preprocessed.shape).astype(np.float32)
    if args.type == "content":
        output = reconstruct(target_features, gen_image, content_loss, args.niter)
    elif args.type == "style":
        output = reconstruct(target_features, gen_image, style_loss, args.niter)
    plt.imshow(output)
    plt.show()


    