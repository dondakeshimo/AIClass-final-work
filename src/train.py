import argparse
import matplotlib.pyplot as plt
from termcolor import cprint
import time

from keras import backend as K
from keras.utils import plot_model

from model import Foreigner_classifier


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


def plot_history(history, dir):
    plt.plot(history.history['acc'], "o-", label="accuracy")
    plt.plot(history.history['val_acc'], "o-", label="accuracy")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.savefig(dir + "acc.png")
    plt.show()

    plt.plot(history.history['loss'], "o-", label="loss",)
    plt.plot(history.history['val_loss'], "o-", label="loss",)
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.savefig(dir + "loss.png")
    plt.show()


def main():
    args = argparser()
    print(args)
    foreign_clf = Foreigner_classifier(args.input_dir_path)
    foreign_clf.make_model()
    # foreign_clf.make_model_from_pre_trained("../data/model")
    plot_model(foreign_clf.model,
               to_file="../data/model.png", show_shapes=True)
    foreign_clf.train_aug_img_model()
    foreign_clf.save_model(args.checkpoint_path)
    foreign_clf.evaluate()
    foreign_clf.show_error_detail()
    plot_history(foreign_clf.history, "../data/history/")
    foreign_clf.save_error_image("../data/err_images/")
    K.clear_session()


if __name__ == "__main__":
    main()
