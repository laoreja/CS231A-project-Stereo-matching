# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image and disparity data to TFRecords file format with Example protos.

The image data set is expected to reside in png files located in the
following directory structure.

    data_dir/label_0/image0.jpeg
    data_dir/label_0/image1.jpg
    ...
    data_dir/label_1/weird-image.jpeg
    data_dir/label_1/my-image.jpeg
    ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

    train_directory/train-00000-of-01024
    train_directory/train-00001-of-01024
    ...
    train_directory/train-01023-of-01024

and

    validation_directory/validation-00000-of-00128
    validation_directory/validation-00001-of-00128
    ...
    validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/colorspace: string, specifying the colorspace, always 'RGB'
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always 'JPEG'

    image/filename: string containing the basename of the image file
                        e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
    image/class/label: integer specifying the index in a classification layer.
        The label ranges from [0, num_labels] where 0 is unused and left as
        the background class.
    image/class/text: string specifying the human-readable version of the label
        e.g. 'dog'

If your data set involves bounding boxes, please look at build_imagenet_data.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
from scipy import misc
from pfm import *
import pickle

#tf.app.flags.DEFINE_string('pickle_path', '/home/laoreja/dataset/SceneFlow/tmp.pickle',
#                                    'Training/Validation list')
tf.app.flags.DEFINE_string('pickle_path', '/home/laoreja/dataset/SceneFlow/train_test_list.pickle',
                           'Training/Validation list')
tf.app.flags.DEFINE_string('png_root', '/home/laoreja/dataset/SceneFlow/frames_cleanpass_png',
                           'Png directory')
tf.app.flags.DEFINE_string('disparity_root', '/home/laoreja/dataset/SceneFlow/disparity',
                           'Disparity directory')
tf.app.flags.DEFINE_string('output_directory', '/home/laoreja/dataset/SceneFlow/tfrecords_SceneFlow',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1024,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 16,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 16,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('height', 540,
                            'Image height.')
tf.app.flags.DEFINE_integer('width', 960,
                            'Image width.')



FLAGS = tf.app.flags.FLAGS

#def _int64_feature(value):
#    """Wrapper for inserting int64 features into Example proto."""
#    if not isinstance(value, list):
#        value = [value]
#    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, relative_dir, left_image, right_image, disparity, mask):
    """Build an Example proto for an example.

    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
    Returns:
        Example proto
    """
    left_image_raw = left_image.tostring()
    right_image_raw = right_image.tostring()
    mask_raw = mask.tostring()
    disparity_raw = disparity.tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature={
            'left_image_raw': _bytes_feature(left_image_raw),
            'right_image_raw': _bytes_feature(right_image_raw),
            'mask_raw': _bytes_feature(mask_raw),
            'disparity_raw': _bytes_feature(disparity_raw),
            'filename': _bytes_feature(tf.compat.as_bytes(filename)),
            'relative_dir': _bytes_feature(tf.compat.as_bytes(relative_dir))}))
    return example

def _process_image(filename, relative_dir):
    """Process a single image file.

    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    left_image_path = os.path.join(FLAGS.png_root, relative_dir, 'left', filename)
    right_image_path = os.path.join(FLAGS.png_root, relative_dir, 'right', filename)
    pfm_name = os.path.splitext(filename)[0]+'.pfm'
    left_disparity_path = os.path.join(FLAGS.disparity_root, relative_dir, 'left', pfm_name)
    right_disparity_path = os.path.join(FLAGS.disparity_root, relative_dir, 'right', pfm_name)
    
    with open(left_disparity_path, 'r') as fd:
        _disparity, _ = load_pfm(fd)
        _disparity = _disparity[::-1, :]
        _disparity = _disparity.astype(np.float32)
    
    with open(right_disparity_path, 'r') as fd:
        _right_disparity, _ = load_pfm(fd)
        _right_disparity = _right_disparity[::-1, :]

    idx = np.tile(np.arange(FLAGS.width), (FLAGS.height, 1))
    tmp = idx - _disparity
    
    _mask = np.ones((FLAGS.height, FLAGS.width), dtype=np.bool)
    _mask[np.where(tmp < 0)] = False
    _mask[np.where(tmp >= FLAGS.width)] = False
    tmp = np.clip(tmp, 0, FLAGS.width-1)
    tmp2 = np.fabs(_disparity - \
                   _right_disparity[
                        np.tile(np.arange(FLAGS.height).reshape((-1, 1)), (1, FLAGS.width)).flatten(), 
                        tmp.round().astype(np.int32).flatten()
                        ].reshape((FLAGS.height, FLAGS.width)))
    _mask[np.where(tmp2 > 1)] = False
    _mask = _mask.astype(np.uint8)

    _left_image = misc.imread(left_image_path)
    _right_image = misc.imread(right_image_path)
    
#    # Check that image converted to RGB
#    assert len(image.shape) == 3
#    height = image.shape[0]
#    width = image.shape[1]
#    assert image.shape[2] == 3

    return _left_image, _right_image, _disparity, _mask


def _process_image_files_batch(thread_index, ranges, name, filenames, relative_dirs, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to
            analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            relative_dir = relative_dirs[i]

            try:
                _left_image, _right_image, _disparity, _mask = _process_image(filename, relative_dir)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected eror while decoding %s, %s.' % (relative_dir, filename))
                print(_left_image.shape, _right_image.shape, _disparity.shape, _mask.shape)
                continue

            example = _convert_to_example(filename, relative_dir, _left_image, _right_image, _disparity, _mask)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                            (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
                    (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
                (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, cnt, record_list, num_shards):
    """Process and save list of images as TFRecord of Example protos.

    Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
#    checked
#    list_sum = sum([len(item['img_list']) for item in record_list])
#    assert(cnt == list_sum)

    filenames = []
    relative_dirs = []
    for record in record_list:
        cur_dir = record['relative_dir']
        for img_name in record['img_list']:
            filenames.append(img_name)
            relative_dirs.append(cur_dir)
    
    assert len(filenames) == len(relative_dirs)
    assert cnt == len(filenames)
            
    
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, cnt, FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, filenames, relative_dirs, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
                (datetime.now(), cnt))
    sys.stdout.flush()


def main(unused_argv):
    global train_cnt
    global test_cnt
    global train_list
    global test_list
    
    with open(FLAGS.pickle_path, 'r') as fd:
        train_cnt, test_cnt, train_list, test_list = pickle.load(fd) 
    
    assert not FLAGS.train_shards % FLAGS.num_threads, (
            'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
            'Please make the FLAGS.num_threads commensurate with '
            'FLAGS.validation_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    # Run it!
    _process_image_files('validation', test_cnt, test_list, FLAGS.validation_shards)
    _process_image_files('train', train_cnt, train_list, FLAGS.train_shards)

if __name__ == '__main__':
    tf.app.run()
    
    
