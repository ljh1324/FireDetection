from PIL import Image
from PIL import ImageFilter
import random
import copy


def composite_fixed_imgs(src_img, obj_img, info, composed_img_name, part_of_img_name, part_of_composite_img_name):
    src_x, src_y = info['src_size']
    src_img = src_img.resize((src_x, src_y))    # set fixed size

    size_ratio = random.randrange(1000, 3000) / 10000

    obj_x = int(info['src_size'][0] * size_ratio)
    obj_y = int(info['src_size'][1] * size_ratio)

    obj_img = obj_img.resize((obj_x, obj_y))    # set fixed size
    obj_img_raw = obj_img.load()                # load obj img pixel data

    loc_x = random.randrange(0, src_x - obj_x)
    loc_y = random.randrange(0, src_y - obj_y)

    part_of_img = src_img.crop((loc_x, loc_y, loc_x + obj_x, loc_y + obj_y)) # crop src image
    part_of_img.save(part_of_img_name)

    copy_img = copy.deepcopy(part_of_img)                                          # copy croped image

    part_of_img_raw = part_of_img.load()
    for i in range(obj_x):
        for j in range(obj_y):
            if not check_range(obj_img_raw[i, j], info['background'], info['theta']):
                part_of_img_raw[i, j] = obj_img_raw[i, j]                          # composite part_of_image, fire

    part_of_img.filter(ImageFilter.MedianFilter)
    part_of_img.save(part_of_composite_img_name)
    part_of_img = Image.open(part_of_composite_img_name)

    blend_img = Image.blend(copy_img, part_of_img, info['alpha'])                 # composite part_of_image, part_of_composite_image with alpha

    src_img.paste(blend_img, (loc_x, loc_y))                                      # composite src_image, composite final part_of_image

    blend_img.save(part_of_composite_img_name)                                    # save
    src_img.save(composed_img_name)

    return src_img, [loc_x, loc_y, obj_x, obj_y]                                 # for saving np data


def check_range(color, background, theta):
    c_r = color[0]
    c_g = color[1]
    c_b = color[2]

    b_r = background[0]
    b_g = background[1]
    b_b = background[2]

    if abs(c_r - b_r) <= theta and abs(c_g - b_g) <= theta and abs(c_b - b_b) <= theta:
        return True
    return False
