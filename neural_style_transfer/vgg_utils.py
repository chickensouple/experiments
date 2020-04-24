import numpy as np
import tensorflow as tf

class VGGLayers(object):
    def __init__(self, input_shape):
        self.vgg = tf.keras.applications.VGG19(include_top=False, input_shape=input_shape, weights='imagenet')

    def get_layer_names(self):
        return [layer.name for layer in self.vgg.layers]

    def get_model(self, layers):
        outputs = [self.vgg.get_layer(layer).output for layer in layers]
        return tf.keras.Model([self.vgg.input], outputs)

def preprocess_image(image, resize=False):
    """
    Preprocess and rgb image that takes on values in the range [0, 255]
    by changing it to bgr and subtracting a mean. 
    This gives the right format to feed into VGG

    Arguments:
        image {np.array} -- Image to be preprocessed.

    Keyword Arguments:
        resize {bool} -- If true, resizes image to (224, 224) (default: {False})

    Returns:
        np.array -- Floating point array for preprocessed image.
    """
    x = np.copy(image)
    if resize:
        x = cv2.resize(x, (224, 224))
    x = np.array(x, dtype=np.float32)
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x

def preprocess_image_inv(preprocessed_image):
    """
    Computes inverse of preprocess_image(). This takes in 
    a preprocessed image, and spits out the original.
    """
    x = np.copy(preprocessed_image)
    mean = [103.939, 116.779, 123.68]
    x = preprocessed_image
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]
    x = x[..., ::-1]
    x = np.array(np.rint(x), dtype=np.uint8)
    return x
