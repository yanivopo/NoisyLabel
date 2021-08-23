import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utils
import config
tf.random.set_seed(1234)
np.random.seed(20)


class Accurecy(Callback):
    def __init__(self, model_cifar, x_test, y_test):
        super(Accurecy, self).__init__()
        self.model_cifar = model_cifar
        self.x_test = x_test
        self.y_test = y_test
        self.hist_acc = []

    def on_epoch_end(self, epoch, logs=None):
        out = self.model_cifar.predict(self.x_test)
        y_pred = np.argmax(out, axis=1)
        acc = np.sum(self.y_test.flatten() == y_pred) / len(y_pred)
        self.hist_acc.append(acc)
        if epoch % 2 == 0:
            print("After {} epochs, accurecy {}".format(epoch, acc))
                                                                                                                                                                                                                                                                                                                                                                  

def plot_graph(data, label, xlabel, ylabel, title, fname=None):
    assert len(data) == len(label), "The number of graph should be equal to number of label"
    plt.figure()
    plt.title(title)
    for i in range(len(data)):
        plt.plot(data[i], label=label[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="lower left")
    if fname is not None:
        plt.savefig("{}_{}".format(fname, title))
    # plt.show()


def plot_training(history, fname=None, txt="Upper bound"):
    data = [history.history["loss"], history.history["val_loss"]]
    label = ["train_loss", "val_loss"]
    title = "{} - Training Loss".format(txt)
    plot_graph(data, label, "Epoch #", "Loss", title, fname)

    data = [history.history["accuracy"], history.history["val_accuracy"]]
    label = ["train_acc", "val_acc"]
    title = "{} - Training Accuracy".format(txt)
    plot_graph(data, label, "Epoch #", "Accuracy", title, fname)


class CifarCNN:
    def __init__(self, activation='relu', num_filters=32, input_shape=(32, 32, 3), class_number=10, drop_dense=0.5,
                 optimizer='adam', epochs=100, batch_size=128):
        self.activation = activation
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.class_number = class_number
        self.drop_dense = drop_dense
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True, width_shift_range=0.1,
                                          height_shift_range=0.1)
        self.model = self.build_model()

    def _conv_block(self, inputs, num_filters=32):
        x = layers.Conv2D(num_filters, (3, 3), activation=self.activation, padding="same")(inputs)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Conv2D(num_filters, (3, 3), activation=self.activation, padding="same")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        return x

    def build_model(self):
        inputs = layers.Input(self.input_shape)

        x = self._conv_block(inputs, self.num_filters)  # reduce to 16x16x32
        x = self._conv_block(x, 2 * self.num_filters)  # reduce to 8x8x64
        x = self._conv_block(x, 4 * self.num_filters)  # reduce to 4x4x128

        x = layers.Flatten()(x)
        x = layers.Dense(512, activation=self.activation)(x)
        x = layers.Dropout(self.drop_dense)(x)
        outputs = layers.Dense(self.class_number, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    def train(self, x_train, y_train, x_test, y_test):
        collback = [tf.keras.callbacks.EarlyStopping(patience=25)]
        self.model.compile(optimizer=self.optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        history = self.model.fit(self.datagen.flow(x_train, y_train, batch_size=self.batch_size), callbacks=collback,
                                 steps_per_epoch=len(x_train) // self.batch_size, epochs=self.epochs,
                                 validation_data=(x_test, y_test))
        return history

    def evaluation(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)
        print("The final accuracy is: {}".format(test_acc))
        return test_loss, test_acc

    def predict(self, x):
        return self.model.predict(x)


class SiameseCifar:
    def __init__(self, input_shape=(32, 32, 3), cifar_model=None, batch_size=128, epochs=30, optimizer='adam',
                 save_model=False, save_dir="temp"):
        self.cifar_class = CifarCNN() if cifar_model is None else cifar_model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.save_model = save_model
        self.save_dir = save_dir
        self.model = self.build_model()

    def build_model(self):
        cifar_model = self.cifar_class.model
        input_a = layers.Input(self.input_shape)
        input_b = layers.Input(self.input_shape)

        out_a = cifar_model(input_a)
        out_b = cifar_model(input_b)
        out_a = layers.Reshape((1, -1))(out_a)
        out_b = layers.Reshape((1, -1))(out_b)
        outputs = layers.Concatenate(axis=1)([out_a, out_b])
        model = models.Model(inputs=[input_a, input_b], outputs=outputs)
        model.summary()
        return model

    @staticmethod
    def my_loss(y_true, y_pred):
        """

        :param y_true: true label with shape (batch_size,1,2)
        :param y_pred: predict label with shape (batch_size,2,class_number)
        :return: The minimum cross-entropy between two optional permutation
        """
        pred_1 = y_pred[:, 0]
        pred_2 = y_pred[:, 1]
        true_1 = y_true[:, 0, 0]
        true_2 = y_true[:, 0, 1]
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        out_1 = cross_entropy(true_1, pred_1) + cross_entropy(true_2, pred_2)
        out_2 = cross_entropy(true_2, pred_1) + cross_entropy(true_1, pred_2)
        out = tf.math.minimum(out_1, out_2) / 2
        return out

    def train(self, x_train, y_train, x_test, y_test, x_valid, y_valid):
        self.model.compile(loss=self.my_loss, optimizer=self.optimizer, metrics=['accuracy'])
        collback = [Accurecy(self.cifar_class.model, x_test, y_test), tf.keras.callbacks.EarlyStopping(patience=25)]
        if self.save_model:
            save_file_name = "_weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
            filepath = os.path.join(self.save_dir, save_file_name)
            collback.append(tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True))

        history = self.model.fit(self.cifar_class.datagen.flow([x_train[:, 0], x_train[:, 1]], y_train,
                                                               batch_size=self.batch_size), epochs=self.epochs,
                                 callbacks=collback, steps_per_epoch=len(x_train) // self.batch_size,
                                 validation_data=([x_valid[:, 0], x_valid[:, 1]], y_valid))
        return history, collback[0].hist_acc

    def evaluation(self, x_test, y_test):
        out = self.cifar_class.predict(x_test)
        y_pred = np.argmax(out, axis=1)
        test_acc = np.sum(y_test.flatten() == y_pred) / len(y_pred)
        print("The final accuracy is: {}".format(test_acc))
        return test_acc


def split_train_valid(x_train, y_train, valid_size=0.2):
    train_number = int(x_train.shape[0] * (1 - valid_size))
    x_valid = x_train[train_number:]
    y_valid = y_train[train_number:]
    x_train = x_train[:train_number]
    y_train = y_train[:train_number]
    return x_train, y_train, x_valid, y_valid


def train_cifar(x_train, y_train, x_test, y_test, param, input_shape=(32, 32, 3), text="upper_bound"):
    cifar = CifarCNN(input_shape=input_shape, epochs=param.epochs, batch_size=param.batch_size,
                     optimizer=param.optimizer, activation=param.activation, drop_dense=param.drop_dense)
    x_train, y_train, x_valid, y_valid = split_train_valid(x_train, y_train, param.valid_size)
    history = cifar.train(x_train, y_train, x_valid, y_valid)
    cifar.evaluation(x_test, y_test)
    plot_training(history, txt=text)
    return cifar


def train_my_model(x_train, y_train, x_test, y_test, param, input_shape=(32, 32, 3)):
    if param.is_save:
        Path(param.save_dir).mkdir(parents=True, exist_ok=True)  # create output dir

    siamese_cifar = SiameseCifar(input_shape=input_shape, epochs=param.epochs, batch_size=param.batch_size,
                                 optimizer=param.optimizer, save_model=param.save_dir)
    x_train, y_train, x_valid, y_valid = split_train_valid(x_train, y_train, param.valid_size)
    hist, acc_list = siamese_cifar.train(x_train, y_train, x_test, y_test, x_valid, y_valid)
    siamese_cifar.evaluation(x_test, y_test)
    plot_training(hist, txt="My_model", fname="graph")

    plot_graph([acc_list], ["train_loss"], "Epoch #", "Accuracy", "Test accuracy")
    # plt.show()


class Parameters:
    def __init__(self, num_pairs_example=5000, valid_size=0.2, activation='relu', optimizer='adam', epochs=100,
                 batch_size=128, drop_dense=0.5, num_filters=32, save_dir="temp", is_save=False):
        self.valid_size = valid_size
        self.num_pairs_example = num_pairs_example
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop_dense = drop_dense
        self.num_filters = num_filters
        self.save_dir = save_dir
        self.is_save = is_save


def main(mode):
    # Get the parameters from the config.py
    param = Parameters(config.NUM_PAIRS_EXAMPLE, config.VALID_SIZE, config.ACTIVATION, config.OPTIMIZER, config.EPOCHS
                       , config.BATCH_SIZE, config.DROP_DENSE, config.NUM_FILTERS, config.SAVE_DIR, config.IS_SAVE)

    example_pairs = param.num_pairs_example
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0  # normalize between 0-1

    img_dim = train_images[0].shape

    # create the new dataset (pair_train_images, pair_train_label_dict) from cifar and also the real label.
    pair_train_images, pair_train_label_dict, real_label = utils.create_pair(train_images, train_labels, example_pairs)

    # plot few examples from the new dataset
    utils.plot_example(pair_train_images, pair_train_label_dict, save=None)

    # convert the train label dictionary to class number
    pair_train_label = np.array([np.nonzero(list(i.values())) for i in pair_train_label_dict])

    # Train one model from ['my_model', 'upper_bound', 'baseline']
    print("Start to train: {}".format(mode))
    if mode == "my_model":
        train_my_model(pair_train_images, pair_train_label, test_images, test_labels, param, img_dim)
    elif mode == "baseline":
        # reshape from (example_pairs,2,img_dim) ----> (2*example_pairs,img_dim), and label to (2*example_pairs,1)
        reshape_train = pair_train_images.reshape(example_pairs * 2, *img_dim)
        reshape_label = pair_train_label.reshape(-1, 1)
        train_cifar(reshape_train, reshape_label, test_images, test_labels, param, img_dim, text="naive")
    else:  # train the upper bound model
        reshape_train = pair_train_images.reshape(example_pairs * 2, *img_dim)
        train_cifar(reshape_train, real_label, test_images, test_labels, param,
                    img_dim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        choices=['my_model', 'upper_bound', 'baseline'],
                        default='my_model',
                        help='choose model to run my_model, upper_bound or baseline')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS.mode)
