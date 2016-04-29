import os
import glob

import numpy as np
import pandas as pd
from skimage import io as sk_io
from skimage import transform as sk_transform
from skimage import color as sk_color


def _load_img(file, img_shape=(64, 64), grayscale=False):
    shape = list(img_shape) + [3]

    img = sk_io.imread(file)
    assert img.shape == (480, 640, 3)

    # crop to right side
    img = sk_transform.resize(img[:, -550:-70], shape)

    if grayscale:
        img = sk_color.rgb2gray(img)
    return img


def load_train_data(
        main_path,
        img_shape=(64, 64),
        swapaxes=True,
        grayscale=False,
        max_img_per_class=None):
    """Load training data

    main_path : path to train/test folder

    img_shape=(64, 64) : desired output image size

    swapaxes=True : if True, reshape to N x Color x Height x Width

    grayscale=False : if True, convert to grayscale

    max_img_per_class=None : if not None, only load first n images per class
    """

    X, y, drivers = [], [], []

    df = pd.read_csv(os.path.join(main_path, 'driver_imgs_list.csv'))
    dct_driver = {img_name: driver for img_name, driver in
                  zip(df['img'], df['subject'])}

    for target in range(10):
        print('Load folder c{}'.format(target))
        path = os.path.join(main_path, 'train', 'c' + str(target), '*.jpg')
        files = glob.glob(path)
        if max_img_per_class:
            files = files[:max_img_per_class]

        for file in files:
            img = _load_img(file, img_shape, grayscale)
            img_name = file.split(os.path.sep)[-1]
            driver = dct_driver[img_name]

            X.append(img)
            y.append(target)
            drivers.append(driver)

    X = np.array(X).reshape(len(X), img_shape[0], img_shape[1], -1)
    X = X.astype(np.float32)
    y = np.array(y).astype(np.int32)
    drivers = np.array(drivers)

    if swapaxes:
        X = np.swapaxes(np.swapaxes(X, 2, 3), 1, 2)

    return X, y, drivers


def load_test_data(
        main_path,
        img_shape=(64, 64),
        swapaxes=True,
        grayscale=False,
        return_ids=False,
        max_img=None):
    """Load test data

    main_path : path to train/test folder

    img_shape=(64, 64) : desired output image size

    swapaxes=True : if True, reshape to N x Color x Height x Width

    grayscale=False : if True, convert to grayscale

    return_ids=False : whether image names should be returned

    max_img=None : if not None, only load first n images
    """

    X = []
    ids = []

    path = os.path.join(main_path, 'test', '*.jpg')
    files = glob.glob(path)
    if max_img:
        files = files[:max_img]

    for file in files:
        img = _load_img(file, img_shape, grayscale)
        X.append(img)

        img_name = file.split(os.path.sep)[-1]
        ids.append(img_name)

    X = np.array(X).reshape(len(X), img_shape[0], img_shape[1], -1)
    X = X.astype(np.float32)
    ids = np.array(ids)

    if swapaxes:
        X = np.swapaxes(np.swapaxes(X, 2, 3), 1, 2)

    return X, ids


def make_submission(fname, y_proba, ids):
    """Make a submission file

    fname : name of file

    y_proba : class probabilities

    ids : image names
    """
    with open(fname, 'w') as f:
        f.write('img,' + ','.join('c' + str(i) for i in range(10)))
        f.write('\n')
        for row, id in zip(y_proba, ids):
            f.write(id + ',')
            f.write(','.join("{:.12f}".format(prob) for prob in row))
            f.write('\n')
