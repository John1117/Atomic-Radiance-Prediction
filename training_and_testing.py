# %%
import numpy as np
import tensorflow as tf
from matplotlib import colors
from keras import layers
from keras.models import Sequential
from data import get_data, shuffle_split
from plot import plot_learning_curve, plot_testing, plot_attribution
from attribution import get_attribution, get_normalized_diff

tf.keras.backend.set_floatx('float64')

# %%
n_data =  300000
n_atom = 32
interparticle_distance = np.linspace(0.1, 1.0, n_data)
intrinsic_decay_rate = 1
dipole_orientation = np.array([1, 0, 0])
wave_vector = np.array([0, 0, 1]) * 2 * np.pi

kernel_image, physical_label = np.zeros((n_data, n_atom, n_atom, 2), dtype=np.float64), np.zeros((n_data, 2 * n_atom), dtype=np.float64)
for i in range(n_data):
    kernel_image[i], physical_label[i] = get_data(n_atom, interparticle_distance[i], intrinsic_decay_rate, dipole_orientation, wave_vector)

# %%
train_interparticle_distance, \
train_kernel_image, \
train_physical_label, \
test_interparticle_distance, \
test_kernel_image, \
test_physical_label = shuffle_split(
    interparticle_distance, 
    kernel_image, 
    physical_label, 
    training_ratio=0.8, 
    training_label_index=np.arange(n_atom)
)
train_physical_label = np.log10(train_physical_label)
test_physical_label = np.log10(test_physical_label)

# %%
with tf.device('/cpu'):
    model = Sequential([
        layers.InputLayer((n_atom, n_atom, 2)),
        layers.Conv2D(4 * n_atom, (n_atom, n_atom), activation='relu', name='conv'),
        layers.Flatten(name='flatten'),
        layers.Dense(2 * n_atom, activation='relu', name='dense1'),
        layers.Dense(2 * n_atom, activation='relu', name='dense2'),
        layers.Dense(n_atom, name='rad'),
    ], name='model')
  
# %%
with tf.device('/cpu'):
    n_epoch = 10
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    history = model.fit(train_kernel_image, train_physical_label, epochs=n_epoch, validation_data=(test_kernel_image, test_physical_label))

plot_learning_curve(n_epoch, history)
plot_testing(interparticle_distance, kernel_image, physical_label, model, np.arange(n_atom))

# %%
baseline = np.concatenate((-0.5 * np.eye(n_atom)[:, :, np.newaxis], np.zeros((n_atom, n_atom, 1))), axis=2)
n_step = 1000
interparticle_distance_arr = np.arange(1/8, 9/8, 1/8)
true_kernel_image_arr = np.zeros((len(interparticle_distance_arr), n_atom, n_atom, 2))
true_physical_label_arr = np.zeros((len(interparticle_distance_arr), 2 * n_atom))
attribution_arr = np.zeros((len(interparticle_distance_arr), n_atom, n_atom, n_atom))
normalized_diff_arr = np.zeros((len(interparticle_distance_arr), n_atom))
for i, interparticle_distance in enumerate(interparticle_distance_arr):
    true_kernel_image, true_physical_label = get_data(n_atom, interparticle_distance, intrinsic_decay_rate, dipole_orientation, wave_vector)
    true_physical_label = np.log10(true_physical_label)
    true_kernel_image_arr[i,:,:,:] = true_kernel_image
    true_physical_label_arr[i,:] = true_physical_label
    for j in range(n_atom):
        attribution_arr[i,j,:,:] = get_attribution(interparticle_distance, model, baseline, true_kernel_image, true_physical_label, label_index=j, n_step=n_step)
        normalized_diff_arr[i,j] = get_normalized_diff(model, true_kernel_image[np.newaxis], true_physical_label, label_index=j)

# %%
for i, interparticle_distance in enumerate(interparticle_distance_arr):
    print(f'\n------------------- ds={interparticle_distance} -------------------')
    max_attribution = np.max(np.abs(attribution_arr[i,:,:,:].flatten()))
    normalized_color = colors.Normalize(-max_attribution, max_attribution)
    for j in range(1, n_atom+1):
        print(f'\n------------------- rad={j} -------------------')
        plot_attribution(interparticle_distance, j, true_kernel_image_arr[i,:,:,:], true_physical_label_arr[i,:], attribution_arr[i,j,:,:], normalized_diff_arr[i,j], normalized_color)