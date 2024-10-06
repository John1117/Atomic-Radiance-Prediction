import numpy as np
import scipy as sp


def get_data(n_atom, interparticle_distance, intrinsic_decay_rate, dipole_orientation, wave_vector):
    atom_index = np.array(range(n_atom))
    atom_position = np.array([[0, 0 ,i * interparticle_distance] for i in atom_index])
    atom_position_diff = np.array([atom_position-atom_position[i] for i in atom_index])
    normalized_atom_distance = np.array([[d/np.linalg.norm(d) if d.any() else np.array([0.0, 0.0, 0.0]) for d in ds] for ds in atom_position_diff])
    wave_number = np.linalg.norm(wave_vector * atom_position_diff, axis=2)

    np.seterr(divide='ignore', invalid='ignore')
    collective_decay_rate = np.nan_to_num(3/2*((1-np.dot(normalized_atom_distance, dipole_orientation)**2)*np.sin(wave_number)/wave_number + (1-3*np.dot(normalized_atom_distance, dipole_orientation)**2)*(np.cos(wave_number)/wave_number**2 - np.sin(wave_number)/wave_number**3)), nan=1.0) #diverge=1.0
    collective_frequency_shift = np.nan_to_num(3/4*(-(1-np.dot(normalized_atom_distance, dipole_orientation)**2)*np.cos(wave_number)/wave_number + (1-3*(np.dot(normalized_atom_distance, dipole_orientation))**2)*(np.sin(wave_number)/wave_number**2 + np.cos(wave_number)/wave_number**3)), nan=0.0) #diverge=0.0
    kernel = (-collective_decay_rate + 2j*collective_frequency_shift) * intrinsic_decay_rate / 2

    kernel_image = np.concatenate((kernel.real[:,:,np.newaxis], kernel.imag[:,:,np.newaxis]), axis=2)

    eigenvalue = np.sort(sp.linalg.eigvals(kernel))[::-1]
    decay_rate, frequency_shift = -2 * eigenvalue.real, eigenvalue.imag
    physical_label = np.concatenate((decay_rate, frequency_shift))

    return kernel_image, physical_label


def shuffle_split(interparticle_distance, kernel_image, physical_label, training_ratio, training_label_index):
    assert len(interparticle_distance) == len(kernel_image) == len(physical_label)
    physical_label = physical_label[:, training_label_index]

    n = len(interparticle_distance)
    p = np.random.permutation(n)
    interparticle_distance, kernel_image, physical_label = interparticle_distance[p], kernel_image[p], physical_label[p]

    split_index = int(np.rint(n * training_ratio))
    return  interparticle_distance[:split_index], \
            kernel_image[:split_index], \
            physical_label[:split_index], \
            interparticle_distance[split_index:], \
            kernel_image[split_index:], \
            physical_label[split_index:]