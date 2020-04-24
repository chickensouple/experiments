import numpy as np
import tensorflow as tf
import imageio
import cv2
import matplotlib.pyplot as plt
from vgg_utils import *
from image_reconstruction import content_loss, style_loss


def style_transfer_loss(style_features, 
    content_features, 
    gen_style_features, 
    gen_content_features,
    content_to_style_ratio):

    loss = content_to_style_ratio * content_loss(gen_content_features, content_features) + \
        style_loss(gen_style_features, style_features)
    return loss

if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser(description="Reconstructing Images from only feature maps.")
    parser.add_argument(
        "--style_image",
        action="store",
        type=str,
        required=True,
        help="Path to the image to use for style.")
    parser.add_argument(
        "--content_image",
        action="store",
        type=str,
        required=True,
        help="Path to the image to use for content.") 
    parser.add_argument(
        "--niter",
        action="store",
        type=int,
        default=1000,
        help="Number of iterations to train for"
    )
    parser.add_argument(
        "--save_file",
        action="store",
        type=str,
        default="images/out.jpg",
        help="File to store generated image in."
    )
    parser.add_argument(
        "--ratio",
        action="store",
        type=float,
        default=1e-3,
        help="The content to style ratio for the loss function. \
              A higher number will weight the content more.")
    args = parser.parse_args()

    # loading images and preprocessing
    style_image = imageio.imread(args.style_image)
    content_image = imageio.imread(args.content_image)
    if len(style_image.shape) == 2:
        style_image = np.stack([style_image for _ in range(3)])
    if style_image.shape[2] == 4:
        style_image = style_image[:, :, :3]
    if len(content_image.shape) == 2:
        content_image = np.stack([content_image for _ in range(3)])
    if content_image.shape[2] == 4:
        content_image = content_image[:, :, :3]
    style_image = preprocess_image(style_image)
    content_image = preprocess_image(content_image)
    # resizing style image to be same size as content image
    style_image = cv2.resize(style_image, 
        (content_image.shape[1], content_image.shape[0]))

    # getting models for different layers
    content_layers = ["block2_conv2", "block4_conv2"]
    style_layers = ["block1_conv1", "block2_conv1",
                    "block3_conv1", "block4_conv1", "block5_conv1"]
    vgg_layers = VGGLayers(content_image.shape)
    layer_model = vgg_layers.get_model(content_layers + style_layers)

    split_idx = len(content_layers)
    content_image = np.array([content_image])
    style_image = np.array([style_image])
    content_features = layer_model(content_image)[:split_idx]
    style_features = layer_model(style_image)[split_idx:]
    
    gen_image = np.random.random(content_image.shape).astype(np.float32)
    gen_image = tf.Variable(gen_image)
    variables = [gen_image]

    initial_lr = 10
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100, 500],
        values=[10, 5, 2]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for i in range(args.niter):
        with tf.GradientTape() as tape:
            gen_features = layer_model(gen_image)
            gen_content_features = gen_features[:split_idx]
            gen_style_features = gen_features[split_idx:]
            loss = style_transfer_loss(style_features, 
                content_features, 
                gen_style_features, 
                gen_content_features,
                args.ratio)
        grads = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grads, variables))
        print("Iteration {}: {}".format(i, loss))

    output = np.array(gen_image[0, ...])
    output = preprocess_image_inv(output)

    imageio.imwrite(args.save_file, output)
    plt.imshow(output)
    plt.show()
