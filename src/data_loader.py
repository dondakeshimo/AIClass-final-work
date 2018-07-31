import glob
import json
import numpy as np
from PIL import Image


img_list = glob.glob("data/JLeaguers/images/*.jpg")
ant_list = [s.replace("images", "annotations") for s in img_list]
ant_list = [s.replace("jpg", "json") for s in ant_list]

img = []
ant = []
for img_path, ant_path in zip(img_list, ant_list):
    img.append(np.array(Image.open(img_path)))

    with open(ant_path, "rt") as f:
        ant_dict = json.load(f)

    ant.append(ant_dict["birthplace"])

len(img)
img[0].shape
np.array(img).shape

ant
