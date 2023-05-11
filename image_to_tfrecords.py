# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image
import rasterio
from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
# The URLs where the MNIST data can be downloaded.
# _DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'motion_data/train'
# _TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 'motion_data/test'
# _TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'

_IMAGE_HEIGHT = 64
_IMAGE_WIDTH = 64
_NUM_CHANNELS = 3


# # The names of the classes.
# _CLASS_NAMES = [
#     'zero',
#     'one',
#     'two',
#     'three',
#     'four',
#     'five',
#     'size',
#     'seven',
#     'eight',
#     'nine',
# ]
def get_filelist(filename):
    return [os.path.join(filename, f) for f in os.listdir(filename)]


def _extract_images(filename):
    """Extract the images into a numpy array.
    Args:
      filename: The path to an MNIST images file.
      num_images: The number of images in the file.
    Returns:
      A numpy array of shape [number_of_images, height, width, channels].
    """
    print('Extracting images from: ', filename)

    image_list = get_filelist(filename)
    data = []
    num_images = len(image_list)
    for image in image_list:

        if image.endswith('DS_Store') or image.endswith('txt'):
            num_images = num_images - 1
            continue
        with rasterio.open(image) as image:
            image_array = image.read()
        img = ToTensor()(image_array)
        img = np.array(img, np.float32).transpose( 2, 0, 1)

        # if len(img.split()) == 1:
        #     img = img.convert("RGB")
        # img = img.resize((self.img_size,self.img_size))

        #label = self.name2label[os.path.basename(file).split('-')[0]]

        #img_resize = np.array(img.resize((_IMAGE_WIDTH, _IMAGE_HEIGHT)))  # 注意顺序，一开始就把宽高顺序写反了，导致图片一致显示有误。


        img = img.astype(np.uint32)
        data.append(img)

    # data = np.array(data).astype(np.uint8)
    # data = data.reshape(num_images, _IMAGE_HEIGHT,_IMAGE_WIDTH,3)

    return data, num_images


def _extract_labels(filename):
    """Extract the labels into a vector of int64 label IDs.
    Args:
      filename: The path to an MNIST labels file.
      num_labels: The number of labels in the file.
    Returns:
      A numpy array of shape [number_of_labels]
    """
    print('Extracting labels from: ', filename)
    image_list = get_filelist(filename)
    labels = []
    num_labels = len(image_list)
    for image in image_list:

        if image.endswith('DS_Store') or image.endswith('txt'):
            num_labels = num_labels - 1
            continue

        label = image.split('/')[-1].split('-')[0].encode('utf-8')
        labels.append(label)
        # print(label)

    return labels, num_labels


def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': bytes_feature(class_id),  # 有更改
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def _add_to_tfrecord(data_filename,
                     tfrecord_writer):
    """Loads data from the binary MNIST files and writes files to a TFRecord.
    Args:
      data_filename: The filename of the MNIST images.
      labels_filename: The filename of the MNIST labels.
      num_images: The number of images in the dataset.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    images, num_images = _extract_images(data_filename)
    labels, num_labels = _extract_labels(data_filename)

    shape = (_IMAGE_HEIGHT, _IMAGE_WIDTH, _NUM_CHANNELS)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_jpeg = tf.image.encode_jpeg(
            image)  # 这里要注意转码格式，因为我的代码是基于tf官方教程改的，是针对mnist图片，png格式的，我自己的训练图片是jpg的，转码不对也会出现问题。

        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
                sys.stdout.flush()

                png_string = sess.run(encoded_jpeg, feed_dict={image: images[j]})

                example = image_to_tfexample(
                    png_string, 'jpg'.encode(), _IMAGE_HEIGHT, _IMAGE_WIDTH, labels[j])  # 这里同上，也要注意转码格式
                tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.
    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.
    Returns:
      An absolute file path.
    """
    return '%s/invoice_%s.tfrecord' % (dataset_dir, split_name)


def run(dataset_dir):
    """Runs the download and conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')

    print(training_filename)
    print(testing_filename)

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # _download_dataset(dataset_dir)

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        data_filename = _TRAIN_DATA_FILENAME
        _add_to_tfrecord(data_filename, tfrecord_writer)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        data_filename = _TEST_DATA_FILENAME
        _add_to_tfrecord(data_filename, tfrecord_writer)

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the invoice  dataset!')


if __name__ == '__main__':
    dataset_dir = './data/tfrecord'
    run(dataset_dir)
    pass


