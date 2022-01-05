import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_filters, n_layers, dim_latent):
        super(Encoder, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.dim_latent = dim_latent

        self.Downsampling = tf.keras.Sequential()
        for i in range(self.n_layers):
            self.Downsampling.add(tf.keras.layers.Conv2D(self.n_filters * (2 ** i),
                                                         kernel_size=(4, 4),
                                                         strides=(2, 2),
                                                         padding='same',
                                                         activation='linear',
                                                         use_bias=False
                                                         )
                                  )
            self.Downsampling.add(tf.keras.layers.BatchNormalization())
            self.Downsampling.add(tf.keras.layers.LeakyReLU())
        self.Downsampling.add(tf.keras.layers.GlobalAveragePooling2D())
        self.Mean = tf.keras.layers.Dense(self.dim_latent,
                                          activation='linear'
                                          )
        self.LogVar = tf.keras.layers.Dense(self.dim_latent,
                                            activation='linear'
                                            )

    def call(self, X):
        X = self.Downsampling(X)
        mean = self.Mean(X)
        logvar = self.LogVar(X)
        return mean, logvar


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_nodes, n_filters, n_layers):
        super(Decoder, self).__init__()
        self.n_nodes = n_nodes
        self.n_filters = n_filters
        self.n_layers = n_layers

        self.FC = tf.keras.layers.Dense(self.n_nodes,
                                        activation='relu',
                                        kernel_initializer='he_normal'
                                        )
        self.Upsampling = tf.keras.Sequential()
        for i in range(self.n_layers - 1):
            self.Upsampling.add(tf.keras.layers.UpSampling2D(size=(2, 2),
                                                             interpolation='nearest'
                                                             )
                                )
            self.Upsampling.add(tf.keras.layers.Conv2D(self.n_nodes / (2 ** i),
                                                       kernel_size=(3, 3),
                                                       activation='linear',
                                                       use_bias=False,
                                                       padding='same'
                                                       )
                                )
            self.Upsampling.add(tf.keras.layers.BatchNormalization())
            self.Upsampling.add(tf.keras.layers.LeakyReLU())
        self.Upsampling.add(tf.keras.layers.UpSampling2D(size=(2, 2),
                                                         interpolation='nearest'
                                                         )
                            )
        self.Upsampling.add(tf.keras.layers.Conv2D(3,
                                                   kernel_size=(3, 3),
                                                   padding='same',
                                                   activation='sigmoid'
                                                   )
                            )

    def call(self, latent):
        latent = self.FC(latent)
        latent = tf.reshape(latent, shape=(-1, 4, 4, 256))
        reconstruction = self.Upsampling(latent)
        return reconstruction


class DFCVAE(tf.keras.models.Model):
    def __init__(self,
                 n_nodes=4096,
                 E_filters=32,
                 E_layers=4,
                 D_filters=128,
                 D_layers=4,
                 dim_latent=100):
        super(DFCVAE, self).__init__()
        self.n_nodes = n_nodes
        self.E_filters = E_filters
        self.E_layers = E_layers
        self.D_filters = D_filters
        self.D_layers = D_layers
        self.dim_latent = dim_latent

        self.vgg = tf.keras.applications.vgg19.VGG19(include_top=False)
        self.DFCModel = tf.keras.Model(inputs=self.vgg.input,
                                       outputs=[self.vgg.layers[6].output,
                                                self.vgg.layers[11].output,
                                                self.vgg.layers[16].output,
                                                self.vgg.layers[21].output]
                                       )
        self.Encoder = Encoder(self.E_filters, self.E_layers, self.dim_latent)
        self.Decoder = Decoder(self.n_nodes, self.D_filters, self.D_layers)

    @tf.function
    def reparameterization(self, mean, logvar):
        epsilon = tf.random.normal(shape=tf.shape(mean),
                                   dtype='float32'
                                   )
        return (epsilon * tf.exp(.5 * logvar)) + mean

    @tf.function
    def compute_DFC(self, X, reconstruction):
        X_output = self.DFCModel(X)
        recon_output = self.DFCModel(reconstruction)
        loss = 0
        for x_, recon_ in zip(X_output, recon_output):
            loss += tf.reduce_mean(tf.square(x_ - recon_)) / 2
        return loss

    def compile(self, optimizer):
        super(DFCVAE, self).compile()
        self.optimizer = optimizer

    @tf.function
    def train_step(self, Input):
        X, _ = Input
        with tf.GradientTape() as tape:
            mean, logvar = self.Encoder(X)
            latent = self.reparameterization(mean, logvar)
            reconstuction = self.Decoder(latent)
            reconstruction_loss = tf.reduce_mean(self.compute_DFC(X, reconstuction))
            kl_loss = tf.reduce_mean(
                -.5 * tf.reduce_sum((1 + logvar - tf.square(mean) - tf.exp(logvar + 1e-8)), axis=1))
            ELBO = reconstruction_loss + kl_loss
        grads = tape.gradient(ELBO, self.Encoder.trainable_variables + self.Decoder.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.Encoder.trainable_variables + self.Decoder.trainable_variables)
        )
        return {'reconstruction_loss': reconstruction_loss, 'kl_loss': kl_loss}

    def call(self, X):
        mean, logvar = self.Encoder(X)
        latent = self.reparameterization(mean, logvar)
        reconstuction = self.Decoder(latent)
        return reconstuction