import glob
import json
import numpy as np
from PIL import Image


def make_path_list(data_path="data/JLeaguers"):
    img_list = glob.glob(data_path + "images/*.jpg")
    ant_list = [s.replace("images", "annotations") for s in img_list]
    ant_list = [s.replace("jpg", "json") for s in ant_list]
    return img_list, ant_list


def load_data(img_list, ant_list):
    img = []
    ant = []
    for img_path, ant_path in zip(img_list, ant_list):
        img.append(np.array(Image.open(img_path)))

        with open(ant_path, "rt") as f:
            ant_dict = json.load(f)

        ant.append(ant_dict["birthplace"])

    img = np.array(img)
    ant = make_country_bin(ant)

    return img, ant


def make_country_bin(ant):
    with open("data/prefecture.txt", "rt") as f:
        prefecture = f.read().split("\n")

    ant_bin = [0 if i in prefecture else 1 for i in ant]
    ant_bin = np.array(ant_bin)
    return ant_bin.reshape(-1, 1)


def main():
    img_list, ant_list = make_path_list()
    img, ant = load_data(img_list, ant_list)
    return img, ant


if __name__ == "__main__":
    main()
