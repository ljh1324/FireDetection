import numpy as np
import random


def load_data(feature_file_name, location_file_name):
    return np.load(feature_file_name), np.load(location_file_name)


def make_train_and_test_set():
    street_feature_file_name = 'data/view_feature/view_feature.npy'
    street_location_file_name = 'data/view_location/view_location.npy'
    fire_feature_file_name = 'data/composed_feature/0.npy'
    fire_location_file_name = 'data/composed_location/0.npy'
    fire_feature_file_name2 = 'data/composed_feature/1.npy'
    fire_location_file_name2 = 'data/composed_location/1.npy'
    fire_feature_file_name3 = 'data/composed_feature/2.npy'
    fire_location_file_name3 = 'data/composed_location/2.npy'
    fire_feature_file_name4 = 'data/composed_feature/3.npy'
    fire_location_file_name4 = 'data/composed_location/3.npy'
    fire_feature_file_name5 = 'data/composed_feature/4.npy'
    fire_location_file_name5= 'data/composed_location/4.npy'

    street_feature, street_location = load_data(street_feature_file_name, street_location_file_name)
    fire_feature, fire_location = load_data(fire_feature_file_name, fire_location_file_name)
    fire_feature2, fire_location2 = load_data(fire_feature_file_name2, fire_location_file_name2)
    fire_feature3, fire_location3 = load_data(fire_feature_file_name3, fire_location_file_name3)
    fire_feature4, fire_location4 = load_data(fire_feature_file_name4, fire_location_file_name4)
    fire_feature5, fire_location5 = load_data(fire_feature_file_name5, fire_location_file_name5)

    train_feature = street_feature[:400]
    train_feature = np.concatenate([train_feature, fire_feature])
    train_feature = np.concatenate([train_feature, fire_feature2])
    train_feature = np.concatenate([train_feature, fire_feature3])
    train_feature = np.concatenate([train_feature, fire_feature4])

    train_location = street_location[:400]
    train_location = np.concatenate([train_location, fire_location])
    train_location = np.concatenate([train_location, fire_location2])
    train_location = np.concatenate([train_location, fire_location3])
    train_location = np.concatenate([train_location, fire_location4])

    train_class = []
    for i in range(400):
        train_class.append([1, 0])  # non-fire
    for i in range(400):
        train_class.append([0, 1])  # fire
    train_class = np.array(train_class)

    test_feature = street_feature[400:450]
    test_feature = np.concatenate([test_feature, fire_feature5[:50]])

    test_location = street_location[400:450]
    test_location = np.concatenate([test_location, fire_location5[:50]])

    test_class = []
    for i in range(50):
        test_class.append([1, 0])   # non-fire
    for i in range(50):
        test_class.append([0, 1])   # fire
    test_class = np.array(test_class)

    """
    for i in range(len(train_feature)):     # shake
        rand_idx = random.randrange(0, len(train_feature))

        temp = train_feature[i]
        train_feature[i] = train_feature[rand_idx]
        train_feature[rand_idx] = temp

        temp = train_location[i]
        train_location[i] = train_location[rand_idx]
        train_location[rand_idx] = temp

        temp = train_class[i]
        train_class[i] = train_class[rand_idx]
        train_class[rand_idx] = temp

    for i in range(len(test_feature)):  # shake
        rand_idx = random.randrange(0, len(test_feature))

        temp = test_feature[i]
        test_feature[i] = test_feature[rand_idx]
        test_feature[rand_idx] = temp

        temp = test_location[i]
        test_location[i] = test_location[rand_idx]
        test_location[rand_idx] = temp

        temp = test_class[i]
        test_class[i] = test_class[rand_idx]
        test_class[rand_idx] = temp
    """

    print(train_feature.shape)
    print(train_location.shape)
    print(train_class.shape)

    print(test_feature.shape)
    print(test_location.shape)
    print(test_class.shape)

    return train_feature, train_location, train_class, test_feature, test_location, test_class