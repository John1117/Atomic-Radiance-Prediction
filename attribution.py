import numpy as np
import tensorflow as tf


def interpolate_image(baseline, image, interpolation_ratio):
    interpolation_ratio = interpolation_ratio[:, np.newaxis, np.newaxis, np.newaxis]
    baseline = np.expand_dims(baseline, axis=0)
    image = np.expand_dims(image, axis=0)
    image_diff = image - baseline
    interpolated_image = baseline + interpolation_ratio * image_diff
    return interpolated_image


def compute_gradient(model, image, label_index):
    image = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(image)
        predicted_label = model(image)[:,label_index]
    return tape.gradient(predicted_label, image)


def integral_approx(gradient):
    gradient = (gradient[:-1] + gradient[1:])/2
    approx = tf.math.reduce_mean(gradient, axis=0)
    return approx


def integrated_gradient(model, baseline, image, label_index, n_step=100):
    interpolation_ratio = np.linspace(0.0, 1.0, n_step+1)
    interpolated_image = interpolate_image(baseline, image, interpolation_ratio)
    path_gradient = compute_gradient(model, interpolated_image, label_index)
    attribution = (integral_approx(path_gradient) * (image - baseline)).numpy()
    return attribution


def four_fold_symmetrize(attribution):
    return (attribution + np.transpose(attribution) + attribution[::-1,::-1] + np.transpose(attribution[::-1,::-1]))/4


def get_attribution(model, baseline, kernel_image, label_index, n_step=100, symmetrize=False):
    attribution = integrated_gradient(model, baseline, kernel_image, label_index, n_step=100).mean(axis=-1)
    if symmetrize:
        return four_fold_symmetrize(attribution)
    else:
        return attribution