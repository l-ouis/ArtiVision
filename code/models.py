

import tensorflow as tf
import hyperparameters as hp


class Basic(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(Basic, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.architecture = [
              ## Add layers here separated by commas.
              tf.keras.layers.Conv2D(32, 3, 1, activation='relu', padding='same', input_shape=(hp.img_size, hp.img_size, 3)),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Conv2D(32, 3, 1, padding="same",
                   activation="relu"),
              tf.keras.layers.MaxPool2D(2),

              tf.keras.layers.Conv2D(128, 3, 1, activation='relu', padding='same'),
              tf.keras.layers.Conv2D(128, 3, 1, padding="same",
                   activation="relu"),
              tf.keras.layers.MaxPool2D(2),

              tf.keras.layers.Conv2D(256, 3, 1, activation='relu', padding='same'),
              tf.keras.layers.Conv2D(256, 3, 1, activation='relu', padding='same'),
              tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

              tf.keras.layers.Conv2D(512, 3, 1, padding="same",
                   activation="relu"),
              tf.keras.layers.Conv2D(512, 3, 1, padding="same",
                   activation="relu"),
          

              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(193, activation='softmax')
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.SparseCategoricalCrossentropy()(labels, predictions)


class Advanced(tf.keras.Model):
    def __init__(self):
        super(Advanced, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        # VGG19 architecture
        # self.vgg19 = [
        #     # Block 1
        #     tf.keras.layers.Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"),
        #     tf.keras.layers.Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"),
        #     tf.keras.layers.MaxPool2D(2, name="block1_pool"),
        #     # Block 2
        #     tf.keras.layers.Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
        #     tf.keras.layers.Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"),
        #     tf.keras.layers.MaxPool2D(2, name="block2_pool"),
        #     # Block 3
        #     tf.keras.layers.Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"),
        #     tf.keras.layers.Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"),
        #     tf.keras.layers.Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"),
        #     tf.keras.layers.Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv4"),
        #     tf.keras.layers.MaxPool2D(2, name="block3_pool"),
        #     # Block 4
        #     tf.keras.layers.Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"),
        #     tf.keras.layers.Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"),
        #     tf.keras.layers.Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"),
        #     tf.keras.layers.Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv4"),
        #     tf.keras.layers.MaxPool2D(2, name="block4_pool"),
        #     # Block 5
        #     tf.keras.layers.Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"),
        #     tf.keras.layers.Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"),
        #     tf.keras.layers.Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"),
        #     tf.keras.layers.Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv4"),
        #     tf.keras.layers.MaxPool2D(2, name="block5_pool")
        # ]

        self.vgg19 = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling="max",
            classifier_activation="softmax",
        )
        
        self.vgg19.trainable = False


        # for layer in self.vgg19:
        #     layer.trainable = False

        self.head = [
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(512, activation="relu"), 
            tf.keras.layers.Dense(193, activation="softmax")
        ]

        # self.vgg19 = tf.keras.Sequential(self.vgg19, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg19(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
      