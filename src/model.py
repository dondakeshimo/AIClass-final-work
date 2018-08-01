import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from termcolor import cprint
import time

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model

import data_loader


EPOCHS = 10
BATCH_SIZE = 16


class Foreigner_classifier():
    def __init__(self, data_dir):
        self.X, self.y = data_loader.main(data_dir)

        self.y = to_categorical(self.y, num_classes=2)

        self.X_train_org, self.X_test_org, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=0)

        self.normalize_image()

    def normalize_image(self):
        self.X_train = self.X_train_org.astype('float32')
        self.X_test = self.X_test_org.astype('float32')
        self.X_train /= 255.0
        self.X_test /= 255.0

        mean = np.mean(self.X_train, axis=(0, 1, 2, 3))
        std = np.std(self.X_train, axis=(0, 1, 2, 3))
        self.X_train = (self.X_train - mean) / (std + 1e-7)
        self.X_test = (self.X_test - mean) / (std + 1e-7)

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
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self):
        self.history = self.model.fit(self.X_train,
                                      self.y_train,
                                      epochs=EPOCHS,
                                      batch_size=BATCH_SIZE,
                                      verbose=1)

    def train_aug_img_model(self):
        datagen = ImageDataGenerator(width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=True)

        self.history = self.model.fit_generator(
            datagen.flow(self.X_train, self.y_train, BATCH_SIZE),
            steps_per_epoch=len(self.X_train) // BATCH_SIZE,
            epochs=EPOCHS)

    def evaluate(self):
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        cprint('Test loss:' + str(loss), "green")
        cprint('Test acc :' + str(acc), "green")

    def show_error_detail(self):
        self.make_error_mask()
        self.count_error()
        cprint("Total test   :" + str(self.error_dict["total_test"]), "cyan")
        cprint("Total error  :" + str(self.error_dict["total_err"]), "cyan")
        cprint("Err Japanese :" + str(self.error_dict["Japanese"]), "cyan")
        cprint("Err foreigner:" + str(self.error_dict["foreigner"]), "cyan")

    def make_error_mask(self):
        pred = self.model.predict(self.X_test)
        pred = (pred > 0.5) * 1.0
        mask = (pred == self.y_test)
        self.mask = ~mask[:, 0]

    def count_error(self):
        cnt = np.sum(self.y_test[self.mask], axis=0)
        self.error_dict = {}
        self.error_dict["Japanese"] = int(cnt[0])
        self.error_dict["foreigner"] = int(cnt[1])
        self.error_dict["total_err"] = len(self.y_test[self.mask])
        self.error_dict["total_test"] = len(self.y_test)

    def save_error_image(self, err_dir):
        images = self.X_test_org[self.mask]
        for i in range(len(images)):
            im = Image.fromarray(images[i])
            im.save(err_dir + str(i) + ".jpg")

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
                        default="../data/",
                        nargs="?",
                        help="input data path")
    parser.add_argument("-p", "--pre_trained_model_path",
                        default=None,
                        nargs="?",
                        help="to load checkpoint h5 file path")
    parser.add_argument("-c", "--checkpoint_path",
                        default="../data/model",
                        nargs="?",
                        help="checkpoint h5 file path")
    return parser.parse_args()


# no use
def time_measure(section, start, elapsed):
    lap = time.time() - start - elapsed
    elapsed = time.time() - start
    cprint("{:22}: {:10.2f}[sec]{:10.2f}[sec]".format(section, lap, elapsed),
           "blue")
    return elapsed


def plot_history(history):
    plt.plot(history.history['acc'], "o-", label="accuracy")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    plt.plot(history.history['loss'], "o-", label="loss",)
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


def main():
    args = argparser()
    print(args)
    foreign_clf = Foreigner_classifier(args.input_dir_path)
    foreign_clf.make_model()
    plot_model(foreign_clf.model,
               to_file="../data/model.png", show_shapes=True)
    # foreign_clf.train_model()
    foreign_clf.train_aug_img_model()
    foreign_clf.save_model(args.checkpoint_path)
    foreign_clf.evaluate()
    foreign_clf.show_error_detail()
    K.clear_session()


if __name__ == "__main__":
    main()
