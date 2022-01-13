import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E= tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64 * (2 ** i),
                                   kernel_size=(5, 5),
                                   strides=(2, 2),
                                   activation='relu',
                                   kernel_initializer='he_normal'
                                   ) for i in range(4)
        ] + [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50,
                                  activation='linear'
                                  )
        ])

    def call(self, X):
        latent = tf.nn.tanh(self.E(X))
        return latent


class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

        self.FC = tf.keras.layers.Dense(1024 * 8 * 8,
                                        activation='relu'
                                        )
        self.TransConv = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(64 * (2 ** (4 - i)),
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding='same',
                                            activation='relu',
                                            kernel_initializer='he_normal'
                                            ) for i in range(4)
        ] + [
            tf.keras.layers.Conv2DTranspose(3,
                                            kernel_size=(5, 5),
                                            padding='same',
                                            strides=(1, 1)
                                            )
        ])

    def call(self, latent):
        latent = self.FC(latent)
        latent = tf.reshape(latent, (-1, 8, 8, 1024))
        reconstruction = tf.nn.tanh(self.TransConv(latent))
        return reconstruction


class Discriminator_z(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator_z, self).__init__()

        self.discriminator = tf.keras.Sequential()
        for i in range(3):
            self.discriminator.add(tf.keras.layers.Dense(16 * (2 ** (3 - i)),
                                                         activation = 'linear',
                                                         kernel_initializer='he_normal'
                                                         )
                                   )
            self.discriminator.add(tf.keras.layers.BatchNormalization())
            self.discriminator.add(tf.keras.layers.ReLU())
        self.discriminator.add(tf.keras.layers.Dense(1,
                                                     activation = 'sigmoid'
                                                     )
                               )

    def call(self, z):
        return self.discriminator(z)


class Discriminator_img(tf.keras.layers.Layer):
    def __init__(self, n_labels):
        super(Discriminator_img, self).__init__()
        self.n_labels = n_labels

        self.resize = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16,
                                   kernel_size=(5, 5),
                                   strides=(2, 2),
                                   padding='same',
                                   activation='linear'
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        self.discriminator = tf.keras.Sequential()
        for i in range(3):
            self.discriminator.add(tf.keras.layers.Conv2D(32 * (2 ** i),
                                                          kernel_size=(5, 5),
                                                          strides=(2, 2),
                                                          padding='same',
                                                          activation='linear'
                                                          )
                                        )
            self.discriminator.add(tf.keras.layers.BatchNormalization())
            self.discriminator.add(tf.keras.layers.ReLU())
        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(1024,
                                                     activation='linear'
                                                     )
                               )
        self.discriminator.add(tf.keras.layers.LeakyReLU(.2))
        self.discriminator.add(tf.keras.layers.Dense(1,
                                                     activation='sigmoid'
                                                     )
                               )

    def call(self, X, y):
        X = self.resize(X)
        y = tf.broadcast_to(tf.reshape(y, (tf.shape(X)[0], 1, 1, self.n_labels)),
                            (tf.shape(X)[0], 64, 64, self.n_labels)
                            )
        return self.discriminator(tf.concat([X, y], axis=-1))


class Caae(tf.keras.models.Model):
    def __init__(self, batch_size, n_labels):
        super(Caae, self).__init__()
        self.batch_size = batch_size
        self.n_labels = n_labels

        self.Encoder = Encoder()
        self.Decoder = Decoder()
        self.Discriminator_z = Discriminator_z()
        self.Discriminator_img = Discriminator_img(self.n_labels)

    def compile(self, optimizer):
        super(Caae, self).compile()
        self.gen_optimizer = optimizer
        self.z_optimizer = optimizer
        self.img_optimizer = optimizer

    @tf.function
    def train_step(self, data):
        X, y = data
        y = tf.cast(y, 'float32')
        fake_label = tf.ones_like(y)
        true_label = tf.zeros_like(y)
        prior = tf.random.uniform(shape=(tf.shape(X)[0], 50),
                                  minval=-1,
                                  maxval=1
                                  )

        with tf.GradientTape(persistent=True) as tape:
            z = self.Encoder(X)
            reconstruction = self.Decoder(tf.concat([z, y], axis=-1))
            disc_z_true = self.Discriminator_z(prior)
            disc_z_fake = self.Discriminator_z(z)
            disc_img_true = self.Discriminator_img(X, y)
            disc_img_fake = self.Discriminator_img(reconstruction,y)
            recon_loss = tf.reduce_mean(
                tf.losses.mae(X, reconstruction)
            )
            tv_loss = self.compute_tvloss(reconstruction)
            fake_z = tf.reduce_mean(
                tf.losses.binary_crossentropy(fake_label, disc_z_fake)
            )
            true_z = tf.reduce_mean(
                tf.losses.binary_crossentropy(true_label, disc_z_true)
            )
            gen_z = tf.reduce_mean(
                tf.losses.binary_crossentropy(true_label, disc_z_fake)
            )
            fake_img = tf.reduce_mean(
                tf.losses.binary_crossentropy(fake_label, disc_img_fake)
            )
            true_img = tf.reduce_mean(
                tf.losses.binary_crossentropy(true_label, disc_img_true)
            )
            gen_img = tf.reduce_mean(
                tf.losses.binary_crossentropy(true_label, disc_img_fake)
            )
            gen_loss = recon_loss + tv_loss * 0.0001 + gen_z * 0.0001 + gen_img * 0.0001
            disc_z_loss = fake_z + true_z
            disc_img_loss = fake_img + true_img
        grads = tape.gradient(gen_loss, self.Encoder.trainable_variables + self.Decoder.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(grads, self.Encoder.trainable_variables + self.Decoder.trainable_variables)
        )
        grads = tape.gradient(disc_z_loss, self.Discriminator_z.trainable_variables)
        self.z_optimizer.apply_gradients(
            zip(grads, self.Discriminator_z.trainable_variables)
        )
        grads = tape.gradient(disc_img_loss, self.Discriminator_img.trainable_variables)
        self.img_optimizer.apply_gradients(
            zip(grads, self.Discriminator_img.trainable_variables)
        )
        return {'reconstruction_loss' : recon_loss,
                'discriminator_z_loss' : disc_z_loss,
                'generation_z_loss' : gen_z,
                'discriminator_img_loss' : disc_img_loss,
                'generation_img_los' : gen_img
                }

    @tf.function
    def compute_tvloss(self, reconstruction):
        tv_loss = ((tf.nn.l2_loss(reconstruction[:,1:,:,:] - reconstruction[:, :127,:,:]) / 128) +
                   (tf.nn.l2_loss(reconstruction[:,:,1:,:] - reconstruction[:,:,:127, :]) / 128)) / self.batch_size
        return tv_loss

    @tf.function
    def call(self, X):
        y = tf.keras.backend.random_bernoulli(shape = (tf.shape(X)[0], self.n_labels), p = .5, dtype='float32')
        y = tf.where(tf.equal(y, 0.), -1., y)
        z = self.Encoder(X)
        reconstruction = self.Decoder(tf.concat([z, y], axis = -1))
        #for build
        disc_z = self.Discriminator_z(z)
        disc_img = self.Discriminator_img(reconstruction,y)
        return reconstruction
