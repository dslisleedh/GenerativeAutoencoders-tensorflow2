import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes, dim_latent, dropout_rate):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dim_latent = dim_latent
        self.dropout_rate = dropout_rate

        self.FC = tf.keras.Sequential()
        for i in range(self.n_layers):
            self.FC.add(tf.keras.layers.Dense(self.n_nodes,
                                              activation = 'relu'
                                              ))
            self.FC.add(tf.keras.layers.Dropout(self.dropout_rate))
        self.FC.add(tf.keras.layers.Dense(self.dim_latent,
                                          activation = 'linear'
                                          ))
        self.Mean = tf.keras.layers.Dense(self.dim_latent,
                                          activation = 'linear'
                                          )
        self.LogVar = tf.keras.layers.Dense(self.dim_latent,
                                            activation = 'linear'
                                            )

    def call(self, X):
        X = self.FC(X)
        mean = self.Mean(X)
        logvar = self.LogVar(X)
        return mean, logvar


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes, dropout_rate, output_dim):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim

        self.FC = tf.keras.Sequential()
        for i in range(self.n_layers):
            self.FC.add(tf.keras.layers.Dense(self.n_nodes,
                                              activation = 'relu',
                                              kernel_initializer = 'he_normal'
                                              ))
            self.FC.add(tf.keras.layers.Dropout(self.dropout_rate))
        self.FC.add(tf.keras.layers.Dense(self.output_dim,
                                          activation = 'sigmoid'
                                          ))

    def call(self, latent):
        reconstruction = self.FC(latent)
        return reconstruction

class VAE(tf.keras.models.Model):
    def __init__(self, n_layers = 2, n_nodes = 500, dropout_rate = .15, latent_dim = 2, output_dim = 784):
        super(VAE, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.Encoder = Encoder(self.n_layers, self.n_nodes, self.latent_dim, self.dropout_rate)
        self.Decoder = Decoder(self.n_layers, self.n_nodes, self.dropout_rate, self.output_dim)

    @tf.function
    def reparameteriation(self, mean, logvar):
        epsilon = tf.random.normal(shape = tf.shape(mean),
                                   dtype = 'float32'
                                   )
        return (epsilon * tf.exp(.5 * logvar)) + mean

    def compile(self, optimizer):
        super(VAE, self).compile()
        self.optimizer = optimizer

    def train_step(self, X):
        with tf.GradientTape() as tape:
            mean, logvar = self.Encoder(X)
            latent = self.reparameteriation(mean, logvar)
            reconstuction = self.Decoder(latent)
            reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(X, reconstuction)
            ) * 28 * 28
            kl_loss = tf.reduce_mean(-.5 * tf.reduce_sum((1 + logvar - tf.square(mean) - tf.exp(logvar + 1e-8)), axis = 1))
            ELBO = reconstruction_loss + kl_loss
        grads = tape.gradient(ELBO, self.Encoder.trainable_variables + self.Decoder.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.Encoder.trainable_variables + self.Decoder.trainable_variables)
        )

        return {'reconstruction_loss' : reconstruction_loss, 'kl_loss' : kl_loss}

    def test_step(self, X):
        mean, logvar = self.Encoder(X)
        latent = self.reparameteriation(mean, logvar)
        reconstuction = self.Decoder(latent)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(X, reconstuction)
        ) * 28 * 28
        kl_loss = tf.reduce_mean(-.5 * tf.reduce_sum((1 + logvar - tf.square(mean) - tf.exp(logvar + 1e-8)), axis=1))
        return {'reconstruction_loss' : reconstruction_loss, 'kl_loss' : kl_loss}
