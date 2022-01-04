from sklearn.datasets import make_swiss_roll
import tensorflow as tf
import numpy as np

class make_priors(object):
    def __init__(self, prior:str, categories = None):
        if prior not in ['swissroll','normal']:
            raise ValueError('priors must be swissroll or normal')
        else:
            self.prior = prior
        self.categories = categories

    def get_samples(self, n_samples:int, dim_latent = 2):
        self.n_samples = n_samples
        self.dim_latent = dim_latent
        if (self.dim_latent > 2) & (self.categories is not None):
            raise ValueError('Only 2 dimensions are supported when return categories')

        if self.prior == 'swissroll':
            return self.swissroll(n_samples)
        else:
            return self.normal(self.dim_latent, self.n_samples)

    def normal(self, dim_latent, n_samples):
        sample = tf.random.normal(shape=(n_samples, dim_latent))
        if self.categories is None:
            return sample
        else:
            y = tf.keras.utils.to_categorical(
                np.digitize(np.arctan2(sample.numpy().T[0],sample.numpy().T[1]), np.linspace(-3.1416, 3.1416, self.categories + 1)) - 1, num_classes = self.categories)
            return tf.stack([sample, y], axis = -1)

    def swissroll(self, n_samples):
        sample = make_swiss_roll(n_samples, noise = .1)
        if self.categories is None:
            return tf.stack([sample[0].T[0], sample[0].T[2]], axis = 1)
        else:
            y = tf.keras.utils.to_categorical(
                np.digitize(sample[1], np.linspace(4.7, 14.3, self.categories + 1)) - 1, num_classes = self.categories)
            return tf.stack([sample[0].T[0], sample[0].T[2], y], axis = -1)