import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import tensorflow as tf


def plot_learning_curve(n_epoch, history):
    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.tick_params(which='major', labelsize=25, pad=10)
    ax1.tick_params(which='minor', labelsize=20, pad=10)
    ax1.tick_params(direction='in', which='major', length=15, width=2)
    ax1.tick_params(direction='in', which='minor', length=7.5, width=1.25)
    ax1.set_xticks(range(1, n_epoch+1))
    ax1.plot(range(1, n_epoch+1), history.history['loss'], 'o-', c=(0,0,0), label='Training', linewidth=2.5)
    ax1.plot(range(1, n_epoch+1), history.history['val_loss'], 'o-', c=(0,0,1), label='Testing', linewidth=2.5)
    ax1.set_yscale('log')
    lg1 = ax1.legend(fontsize=25, edgecolor=(0,0,0), borderaxespad=1)
    lg1.get_frame().set_linewidth(2)
    ax1.set_xlabel('Epoch', fontsize=35)
    ax1.set_ylabel('MSE', fontsize=35)
    plt.show()


def plot_testing(interparticle_distance, true_kernel_image, true_physical_label, model, n_atom):
    with tf.device('/cpu'):
        predicted_physical_label = model.predict(true_kernel_image)

    fig, ax1 = plt.subplots(figsize=(32,18))
    ax1.tick_params(labelsize=35, pad=10)
    ax1.tick_params(direction='in', which='major', length=15, width=2)
    ax1.tick_params(direction='in', which='minor', length=7.5, width=1.25)
    ax1.plot(interparticle_distance[::], true_physical_label[::,:n_atom], '-', c=(0,0,0,0.5), linewidth=1)
    ax1.plot(interparticle_distance[::3000], 10**predicted_physical_label[::3000,:n_atom], '.', c=(0,0,1,1), markersize=10)
    ax1.set_xlabel(r'$d_s/\lambda$', fontsize=40)
    ax1.set_ylabel(r'$\Gamma_m/\Gamma$', fontsize=40)
    ax1.set_xticks(np.arange(0.1,1.1,0.1))

    ax2 = fig.add_axes([0.5875, 0.45, 0.3, 0.4])
    ax2.tick_params(labelsize=25, pad=10)
    ax2.tick_params(direction='in', which='major', length=15, width=2)
    ax2.tick_params(direction='in', which='minor', length=10, width=2)
    ax2.set_xticks(np.arange(0.1,0.4,0.05))
    ax2.plot(interparticle_distance[:100000:], true_physical_label[:100000:,:n_atom], '-', c=(0,0,0,0.5), linewidth=0.5)
    ax2.plot(interparticle_distance[:100000:3000], 10**predicted_physical_label[:100000:3000,:n_atom], '.', c=(0,0,1,1), markersize=5)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$d_s/\lambda$', fontsize=30)
    ax2.set_ylabel(r'$\Gamma_m/\Gamma$', fontsize=30)

    rel_diff = np.abs(true_physical_label[::,:n_atom]-10**predicted_physical_label[::,:n_atom])/true_physical_label[::,:n_atom]
    
    ax3 = fig.add_axes([0.275, 0.55, 0.25, 0.3])
    ax3.tick_params(labelsize=25, pad=10)
    ax3.tick_params(direction='in', which='major', length=15, width=2)
    ax3.tick_params(direction='in', which='minor', length=7.5, width=1.25)
    ax3.fill_between(interparticle_distance[::], np.min(rel_diff, axis=1), np.max(rel_diff, axis=1), color=(1,0,0,0.2))
    ax3.plot(interparticle_distance[::], np.mean(rel_diff, axis=1), '.-', c=(1,0,0,1), markersize=0.05)
    ax3.set_xticks(np.arange(0.1,1.1,0.2))
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$d_s/\lambda$', fontsize=30)
    ax3.set_ylabel('Relative diff.', fontsize=30)
    ax3.set_ylim(10**-6.5, 10**1.5)

    plt.show()


def plot_attribution(interparticle_distance, atom_index, true_kernel_image, true_physical_label, attribution, normalized_diff, relative_normalized_color):
    image_size = attribution.shape[0]
    plt.figure(figsize=(20, 5))
    plt.suptitle('ds={0}, xi={1}pi, rad({2})={3}, nmlzd dif={4}'.format(interparticle_distance, interparticle_distance*2, atom_index+1, true_physical_label[atom_index], normalized_diff), fontsize=20)

    max_element = np.max(np.abs(true_kernel_image.flatten()))
    kernel_normalized_color = colors.Normalize(-max_element, max_element)

    plt.subplot(1,4,1)
    plt.title('knl(real)'.format(interparticle_distance), fontsize=20)
    plt.xticks(np.arange(image_size), np.arange(1,image_size+1))
    plt.yticks(np.arange(image_size), np.arange(image_size,0,-1))
    plt.imshow(true_kernel_image[:,:,0], cmap='bwr', norm=kernel_normalized_color)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=10)

    plt.subplot(1,4,2)
    plt.title('knl(imag.)'.format(interparticle_distance), fontsize=20)
    plt.xticks(np.arange(image_size), np.arange(1,image_size+1))
    plt.yticks(np.arange(image_size), np.arange(image_size,0,-1))
    plt.imshow(true_kernel_image[:,:,1], cmap='bwr', norm=kernel_normalized_color)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=10)

    plt.subplot(1,4,3)
    plt.title('rel atrbu.', fontsize=20)
    plt.xticks(np.arange(image_size), np.arange(1,image_size+1))
    plt.yticks(np.arange(image_size), np.arange(image_size,0,-1))
    plt.imshow(attribution, cmap='bwr', norm=relative_normalized_color)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=10)

    max_attribution = np.max(np.abs(attribution.flatten()))
    absolute_normalized_color = colors.Normalize(-max_attribution, max_attribution)

    plt.subplot(1,4,4)
    plt.title('abs atrbu.', fontsize=20)
    plt.xticks(np.arange(image_size), np.arange(1,image_size+1))
    plt.yticks(np.arange(image_size), np.arange(image_size,0,-1))
    plt.imshow(attribution, cmap='bwr', norm=absolute_normalized_color)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=10)
    plt.show()


def get_normalized_diff(model, true_kernel_image, true_physical_label, label_index):
    with tf.device('/cpu'):
        predicted_physical_label = model.predict(true_kernel_image)[0]
    return (((10**predicted_physical_label[label_index] - 10**true_physical_label[label_index])**2).mean())**0.5 / 10**true_physical_label[label_index]