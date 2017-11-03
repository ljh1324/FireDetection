import os
import copy
import numpy as np
from useful_function import join

from make_composite_image import composite_fixed_imgs

from PIL import Image


def img_to_numpy(img):
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

    return feature


def composite_fixed_imgs_in_dirs(src_dir, obj_dir, info, composed_dir, part_of_imgs_dir, part_of_composite_imgs_dir, features_dir, location_dir):
    src_files = os.listdir(src_dir)
    obj_files = os.listdir(obj_dir)

    if not os.path.exists(composed_dir):
        os.makedirs(composed_dir)
    if not os.path.exists(part_of_imgs_dir):
        os.makedirs(part_of_imgs_dir)
    if not os.path.exists(part_of_composite_imgs_dir):
        os.makedirs(part_of_composite_imgs_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(location_dir):
        os.makedirs(location_dir)

    src_images = []
    for file in src_files:
        img = Image.open(join(src_dir, file))
        img = img.convert("RGB")
        src_images.append(img)

    obj_images = []
    for file in obj_files:
        img = Image.open(join(obj_dir, file))
        img = img.convert("RGB")
        obj_images.append(img)

    idx = 1

    feature_list = []
    location_list = []

    idx = 0
    for i in range(len(src_images)):
        for j in range(len(obj_images)):
            # def composite_with_img(src, obj, info, composed_img_name, part_of_img_name, part_of_composite_img_name)
            src_img = copy.deepcopy(src_images[i])
            obj_img = copy.deepcopy(obj_images[j])

            composite_img, location = composite_fixed_imgs(src_img, obj_img, info
                                                           , join(composed_dir, str(idx) + '.jpg'), join(part_of_imgs_dir, str(idx) + '.jpg'), join(part_of_composite_imgs_dir, str(idx) + '.jpg'))

            feature = img_to_numpy(composite_img)
            feature_list.append(feature)
            location_list.append(np.array(location))
            if len(feature_list) >= 100:
                feature_list = np.array(feature_list)
                np.save(join(features_dir, str(idx) + '.npy'), feature_list)
                location_list = np.array(location_list)
                np.save(join(location_dir, str(idx) + '.npy'), location_list)
                feature_list = []
                location_list = []
                idx += 1

    if len(feature_list) > 0:
        feature_list = np.array(feature_list)
        np.save(join(features_dir, str(idx) + '.npy'), feature_list)
        location_list = np.array(location_list)
        np.save(join(location_dir, str(idx) + '.npy'), location_list)


if __name__ == '__main__':
    src_dir = 'data/view'
    obj_dir = 'data/fire'
    composed_dir = 'data/composed'
    part_of_imgs_dir = 'data/part_of_img'
    feature_file_name = 'data/composed_feature/'
    location_file_name = 'data/composed_location/'
    part_of_composite_imgs_dir = 'data/part_of_composed_img'

    info = {'background': (0, 0, 0), 'theta': 20, 'alpha': 0.65, 'src_size': (256, 256)}

    composite_fixed_imgs_in_dirs(src_dir, obj_dir, info, composed_dir, part_of_imgs_dir, part_of_composite_imgs_dir, feature_file_name, location_file_name)