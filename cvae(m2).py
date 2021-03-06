import tensorflow as tf

class AuxClassifier(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes, dropout_rate, n_categories):
        super(AuxClassifier, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout_rate = dropout_rate
        self.n_categories = n_categories

        self.FC =  tf.keras.Sequential()
        for i in range(self.n_layers):
            self.FC.add(tf.keras.layers.Dense(self.n_nodes,
                                              activation = 'relu',
                                              kernel_initializer = 'he_normal'
                                              ))
            self.FC.add(tf.keras.layers.Dropout(self.dropout_rate))
        self.FC.add(tf.keras.layers.Dense(self.n_categories,
                                          activation = 'softmax'
                                          ))

    def call(self, X):
        return self.FC(X)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes, latent_dims, dropout_rate):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.latent_dims = latent_dims
        self.dropout_rate = dropout_rate

        self.FC = tf.keras.Sequential()
        for i in range(self.n_layers):
            self.FC.add(tf.keras.layers.Dense(self.n_nodes,
                                              activation = 'relu',
                                              kernel_initializer = 'he_normal'
                                              ))
            self.FC.add(tf.keras.layers.Dropout(self.dropout_rate))
        self.Mean = tf.keras.layers.Dense(self.latent_dims,
                                          activation='linear'
                                          )
        self.LogVar = tf.keras.layers.Dense(self.latent_dims,
                                            activation='linear'
                                            )

    def call(self, X, y):
        X = self.FC(tf.concat([X, y], axis = -1))
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

    def call(self, latent, y):
        reconstruction = self.FC(tf.concat([latent, y], axis = -1))
        return reconstruction

class CVAE(tf.keras.models.Model):
    def __init__(self, n_layers = 2, n_nodes = 500, dropout_rate = .15, latent_dim = 2, output_dim = 784, n_categories = 10):
        super(CVAE, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_categories = n_categories

        self.AuxClassifier = AuxClassifier(self.n_layers, self.n_nodes, self.dropout_rate, self.n_categories)
        self.Encoder = Encoder(self.n_layers, self.n_nodes, self.latent_dim, self.dropout_rate)
        self.Decoder = Decoder(self.n_layers, self.n_nodes, self.dropout_rate, self.output_dim)

    @tf.function
    def reparameteriation(self, mean, logvar):
        epsilon = tf.random.normal(shape = tf.shape(mean),
                                   dtype = 'float32'
                                   )
        return (epsilon * tf.exp(.5 * logvar)) + mean

    def compile(self, optimizer):
        super(CVAE, self).compile()
        self.optimizer = optimizer

    @tf.function
    def train_step(self, X):
        with tf.GradientTape() as tape:
            y = self.AuxClassifier(X)
            mean, logvar = self.Encoder(X, y)
            latent = self.reparameteriation(mean, logvar)
            reconstuction = self.Decoder(latent, y)
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

    @tf.function
    def test_step(self, X):
        y = self.AuxClassifier(X)
        mean, logvar = self.Encoder(X, y)
        latent = self.reparameteriation(mean, logvar)
        reconstuction = self.Decoder(latent, y)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(X, reconstuction)
        ) * 28 * 28
        kl_loss = tf.reduce_mean(-.5 * tf.reduce_sum((1 + logvar - tf.square(mean) - tf.exp(logvar + 1e-8)), axis=1))
        return {'reconstruction_loss' : reconstruction_loss, 'kl_loss' : kl_loss}
