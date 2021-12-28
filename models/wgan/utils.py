import numpy as np


def compute_derivative(data):
    """(T, D)"""
    return data[1:] - data[:-1]

def compute_velocity(motion):
    '''(T, D)'''
    T = motion.shape[0]
    motion = motion.reshape(T, -1, 3)
    velocity = np.ndarray(shape=(0, motion.shape[1]))
    for t in range(1, T):
        vel = np.sqrt(np.sum((motion[t] - motion[t-1]) ** 2, axis=-1))
        velocity = np.append(velocity, vel[np.newaxis, :], axis=0)
    return velocity