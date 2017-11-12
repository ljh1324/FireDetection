from PIL import Image
from useful_function import join

import numpy as np
import os


# For extract feature from View Pic
def img_to_numpy(img_file, size):
    img = Image.open(img_file)

    src_x, src_y = size

    img = img.convert("RGB")
    img = img.resize((src_x, src_y))
    x, y = img.size
    img_raw = img.load()
    feature = []
    for i in range(x):
        for j in range(y):
            feature.append(img_raw[i, j][0])
            feature.append(img_raw[i, j][1])
            feature.append(img_raw[i, j][2])
    feature = np.array(feature)
    feature = np.reshape(feature, (x, y, 3))

    return img, feature


def make_fixed_image_feature_in_dir(dir, size, save_img_dir, feature_file_name, location_file_name):
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    if not os.path.exists(os.path.dirname(feature_file_name)):
        os.makedirs(os.path.dirname(feature_file_name))

    if not os.path.exists(os.path.dirname(location_file_name)):
        os.makedirs(os.path.dirname(location_file_name))

    images = os.listdir(dir)

    feature_list = []
    location_list = []

    idx = 1
    for image in images:
        img, feature = img_to_numpy(join(dir, image), size)
        img.save(join(save_img_dir, str(idx) + '.jpg'))
        feature_list.append(feature)
        location_list.append([0, 0, 0, 0])                      # non-fired
        idx += 1

    feature_list = np.array(feature_list)
    location_list = np.array(location_list)

    np.save(feature_file_name, feature_list)
    np.save(location_file_name, location_list)


if __name__ == '__main__':
    dir = 'data/view'
    save_img_dir = 'data/resized_view'
    feature_file_name = 'data/view_feature/view_feature.npy'
    location_file_name = 'data/view_location/view_location.npy'
    size = (256, 256)
    make_fixed_image_feature_in_dir(dir, size, save_img_dir, feature_file_name, location_file_name)
