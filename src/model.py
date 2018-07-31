import argparse
from termcolor import cprint
import time

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model


EPOCHS = 100
BATCH_SIZE = 64


class Cifar_classifier():
    def __init__(self, dir_path):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = \
            cifar10.load_data()

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255.0
        self.X_test /= 255.0

        self.y_train = to_categorical(self.y_train, num_classes=10)
        self.y_test = to_categorical(self.y_test, num_classes=10)

    def make_model_from_pre_trained(self, pre_trained_model_path):
        with open(pre_trained_model_path + ".json", "rt")as f:
            json_model = f.read()
        self.model = model_from_json(json_model)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.load_weights(pre_trained_model_path + ".h5")

    def make_model(self, pre_trained_model_path=None):
        self.model = Sequential()

        self.model.add(Conv2D(filters=16, kernel_size=3, padding="same",
                              activation="relu",
                              input_shape=self.X_train.shape[1:]))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Conv2D(filters=32, kernel_size=3, padding="same",
                              activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Conv2D(filters=64, kernel_size=2, padding="same",
                              activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=2, padding="same",
                              activation="relu"))
        self.model.add(Conv2D(filters=32, kernel_size=2, padding="same",
                              activation="relu"))
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.X_train,
                       self.dtrain.target.values,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       verbose=1)

    def evaluate(self):
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        cprint('Test loss:' + str(loss), "green")
        cprint('Test acc :' + str(acc), "green")

    def save_model(self, checkpoint_path):
        self.model.save_weights(checkpoint_path + ".h5")
        with open(checkpoint_path + ".json", "w") as f:
            f.write(self.model.to_json())


# no use
def get_callbacks(checkpoint_path, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(checkpoint_path, save_best_only=True)
    return [es, msave]


def argparser():
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument("-m", "--mode",
                        default=None,
                        nargs="?",
                        help="train or predict")
    parser.add_argument("-i", "--input_dir_path",
                        default="./data/",
                        nargs="?",
                        help="input data path")
    parser.add_argument("-p", "--pre_trained_model_path",
                        default=None,
                        nargs="?",
                        help="to load checkpoint h5 file path")
    parser.add_argument("-c", "--checkpoint_path",
                        default="./data/save_test",
                        nargs="?",
                        help="checkpoint h5 file path")
    parser.add_argument("-s", "--submission_path",
                        default="./data/NNsubmission.csv",
                        nargs="?",
                        help="csv file path for submission")
    return parser.parse_args()


# no use
def time_measure(section, start, elapsed):
    lap = time.time() - start - elapsed
    elapsed = time.time() - start
    cprint("{:22}: {:10.2f}[sec]{:10.2f}[sec]".format(section, lap, elapsed),
           "blue")
    return elapsed


def main():
    args = argparser()
    print(args)
    cifar = Cifar_classifier(args.input_dir_path)
    cifar.make_model()
    plot_model(cifar.model, to_file="./data/model.png", show_shapes=True)
    cifar.train_model()
    cifar.save_model(args.checkpoint_path)
    cifar.evaluate()
    K.clear_session()


if __name__ == "__main__":
    main()
