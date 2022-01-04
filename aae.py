import tensorflow as tf
from priors import make_priors

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes, latent_dims, noise = None, noise_rate = .2):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.latent_dims = latent_dims
        if noise not in [None, 'gaussian', 'dropout']:
            raise ValueError('noise must be None, gaussian, or dropout')
        else:
            self.noise = noise
        self.noise_rate = noise_rate

        if self.noise == 'gaussian':
            self.InputNoise = tf.keras.layers.GaussianNoise(stddev= self.noise_rate)
        elif self.noise == 'dropout':
            self.InputNoise = tf.keras.layers.Dropout(rate = noise_rate)
        else:
            self.InputNoise = tf.keras.layers.Layer()
        self.FC = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_nodes,
                                  activation = 'relu',
                                  kernel_initializer = tf.keras.initializers.random_normal(stddev = .01)
                                  ) for _ in range(self.n_layers)
        ])
        self.FC.add(tf.keras.layers.Dense(self.latent_dims,
                                          activation = 'linear',
                                          kernel_initializer = tf.keras.initializers.random_normal(stddev = .01)
                                          )
                    )

    def call(self, X):
        y = self.InputNoise(X)
        y = self.FC(y)
        return y


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes, output_dims):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.output_dims = output_dims

        self.FC = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_nodes,
                                  activation = 'relu',
                                  kernel_initializer = tf.keras.initializers.random_normal(stddev = .01)
                                  ) for _ in range(self.n_layers)
        ])
        self.FC.add(tf.keras.layers.Dense(output_dims,
                                          activation = 'sigmoid'
                                          ))

    def call(self, X):
        y = self.FC(X)
        return y

class Discriminator(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        self.FC = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_nodes,
                                  activation = 'relu',
                                  kernel_initializer = tf.keras.initializers.random_normal(stddev = .01)
                                  )
        ])
        self.FC.add(tf.keras.layers.Dense(1,
                                          activation = 'sigmoid'
                                          )
                    )

    def call(self, X):
        y = self.FC(X)
        return y


class AAE(tf.keras.models.Model):
    def __init__(self,
                 prior,
                 n_layers = 2,
                 n_nodes = 1000,
                 latent_dims = 2,
                 output_dims = 784,
                 noise = None,
                 noise_rate = .2):
        super(AAE, self).__init__()
        self.prior = prior
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.noise = noise
        self.noise_rate = noise_rate

        self.Encoder = Encoder(self.n_layers, self.n_layers, self.latent_dims, noise = self.noise, noise_rate = self.noise_rate)
        self.Decoder = Decoder(self.n_layers, self.n_nodes, self.output_dims)
        self.Discriminator = Discriminator(self.n_layers, self.n_nodes)
        self.prior_generator = make_priors(self.prior)

    def compile(self, r_optimizer, d_optimizer, g_optimizer):
        super(AAE, self).compile()
        self.r_optimizer = r_optimizer
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @tf.function
    def train_step(self, X):

        #1 Reconstruction
        with tf.GradientTape() as tape:
            latent = self.Encoder(X)
            reconstruction = self.Decoder(latent)
            R_loss = tf.reduce_mean(
                tf.keras.losses.mse(X, reconstruction)
            ) * self.output_dims
        grads = tape.gradient(R_loss,
                              self.Encoder.trainable_variables + self.Decoder.trainable_variables
                              )
        self.r_optimizer.apply_gradients(
            zip(grads,
                self.Encoder.trainable_variables + self.Decoder.trainable_variables)
        )

        #2 Discriminator
        latent = self.Encoder(X)
        prior = self.prior_generator.get_samples(n_samples = tf.shape(X)[0], dim_latent = self.latent_dims)
        with tf.GradientTape() as tape:
            Output_fake = self.Discriminator(latent)
            Output_true = self.Discriminator(prior)
            F_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.ones(shape = tf.shape(Output_fake)), Output_fake)
            )
            T_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros(shape = tf.shape(Output_true)), Output_true)
            )
            D_loss = F_loss + T_loss

        grads = tape.gradient(D_loss,
                              self.Discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(grads,
                self.Discriminator.trainable_variables
                )
        )

        #3 Generator
        with tf.GradientTape() as tape:
            latent = self.Encoder(X)
            G_output = self.Discriminator(latent)
            G_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros(shape = tf.shape(G_output)), G_output)
            )
        grads = tape.gradients(G_loss,
                              self.Encoder.trainable_variables)
        self.g_optimizer.apply_gradient(
            zip(grads,
                self.Encoder.trainable_variables
                )
        )
        return {'reconstruction_loss' : R_loss, 'discrimination_loss' : D_loss, 'generation_loss' : G_loss}
