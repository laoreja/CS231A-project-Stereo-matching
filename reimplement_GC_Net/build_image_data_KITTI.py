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

The images' paths in the KITTI datasets follow strict rules, so I don't use a filelist to build this dataset.

This TensorFlow script converts the training and testing data into
a sharded data set consisting of TFRecord files

  output_directory/train-00000-of-00008
  output_directory/train-00001-of-00008
  ...
  output_directory/train-00007-of-00008

and

  output_directory/test-00000-of-00008
  output_directory/test-00001-of-00008
  ...
  output_directory/test-00007-of-00008

where we have selected 8 and 8 shards for each sub data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:
  
for both training and testing:
  left_image_raw: string containing encoded image in RGB colorspace
  right_image_raw: string containing encoded image in RGB colorspace
  height: integer, image height in pixels
  width: integer, image width in pixels

training only:
  disparity_raw: string containing float formatted grount-truth disparity
  mask_raw: string containing unit8 formatted grount-truth disparity mask

testing only:
  filename: string containing the basename of the image file
            following the KITTI naming rule, is 'xxxxxx_10.png'
  subset_idx: integer, indicating which subset, 0 for KITTI(2012), 1 for KITTI2015
  
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
from uint16png import *

tf.app.flags.DEFINE_string('output_directory', '/home/laoreja/dataset/tfrecords_KITTI_merged',
               'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 8,
              'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
              'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
              'Number of threads to preprocess the images.')


FLAGS = tf.app.flags.FLAGS

num_sub_datasets = 2
left_paths = ['colored_0', 'image_2']
right_paths = ['colored_0', 'image_3']
disparity_paths = ['disp_occ', 'disp_occ_0']
filename_suffix = '_10.png'
train_roots = ['/home/laoreja/stereo/mc-cnn/data.kitti/unzip/training', '/home/laoreja/stereo/mc-cnn/data.kitti2015/unzip/training']
test_roots = ['/home/laoreja/stereo/mc-cnn/data.kitti/unzip/testing', '/home/laoreja/stereo/mc-cnn/data.kitti2015/unzip/testing']

#def _int64_feature(value):
#    """Wrapper for inserting int64 features into Example proto."""
#    if not isinstance(value, list):
#        value = [value]
#    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, subset_idx, left_image, right_image, disparity=None, mask=None):
  """Build an Example proto for an example.

  Args:
    filename: string, image file name, e.g., '000000_10.png'
    subset_idx: integer, indicating which subset, 0 for KITTI(2012), 1 for KITTI2015
    left_image, right_image: string, JPEG encoding of RGB image
    disparity_raw: string, containing float formatted grount-truth disparity
    mask_raw: string, containing unit8 formatted grount-truth disparity mask
  Returns:
    Example proto
  """
  left_image_raw = left_image.tostring()
  right_image_raw = right_image.tostring()
  if disparity is not None:
    mask_raw = mask.tostring()
    disparity_raw = disparity.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(left_image.shape[0]),
        'width': _int64_feature(left_image.shape[1]),
        'left_image_raw': _bytes_feature(left_image_raw),
        'right_image_raw': _bytes_feature(right_image_raw),
        'mask_raw': _bytes_feature(mask_raw),
        'disparity_raw': _bytes_feature(disparity_raw),
        'filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'subset_idx': _int64_feature(subset_idx)
        }))
  else:
    example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(left_image.shape[0]),
      'width': _int64_feature(left_image.shape[1]),
      'left_image_raw': _bytes_feature(left_image_raw),
      'right_image_raw': _bytes_feature(right_image_raw),
      'filename': _bytes_feature(tf.compat.as_bytes(filename)),
      'subset_idx': _int64_feature(subset_idx)
      }))
  return example


def _process_image(filename, subset_idx, is_test):
  """Process a single image file.

  Args:
    filename: string, image file name, e.g., '000000_10.png'
    subset_idx: integer, indicating which subset, 0 for KITTI(2012), 1 for KITTI2015
  Returns:
    _left_image, _right_image: image numpy array of shape [height, width, 3], loaded using misc.read
    _disparity, _mask: numpy array of shape [height, width]

  """
  root_path = test_roots[subset_idx] if is_test else train_roots[subset_idx]
  left_image_path = os.path.join(root_path, left_paths[subset_idx], filename)
  right_image_path = os.path.join(root_path, right_paths[subset_idx], filename)    
  _left_image = misc.imread(left_image_path)
  _right_image = misc.imread(right_image_path)
  
  #  Check that image converted to RGB, left and right image have the same shape
  assert len(_left_image.shape) == 3
  assert _left_image.shape[2] == 3
  assert np.all(_left_image.shape == _right_image.shape)
  
  if is_test:
    return _left_image, _right_image
  
  disparity_image_path = os.path.join(root_path, disparity_paths[subset_idx], filename)
  _disparity, _mask = load_uint16PNG(disparity_image_path)

  return _left_image, _right_image, _disparity, _mask


def _process_image_files_batch(thread_index, ranges, name, cnts, roots, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the sub data set
    cnts: [num_img_pairs_in_KITTI, num_img_pairs_in_KITTI2015]
    roots: data root paths for KITTI and KITTI2015
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
      if i < cnts[0]:
        subset_idx = 0
        filename = ('%06d' % i) + filename_suffix
      else:
        subset_idx = 1
        filename = ('%06d' % (i - cnts[0])) + filename_suffix

      try:
        if name == 'test':
          _left_image, _right_image = _process_image(filename, subset_idx, name=='test')
        else:
          _left_image, _right_image, _disparity, _mask = _process_image(filename, subset_idx, name=='test')
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s, %s, %s.' % (filename, subset_idx, name))
        print(_left_image.shape, _right_image.shape, _disparity.shape, _mask.shape)
        continue

      if name == 'test':
        example = _convert_to_example(filename, subset_idx, _left_image, _right_image)
      else:
        example = _convert_to_example(filename, subset_idx, _left_image, _right_image, _disparity, _mask)
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


def _process_image_files(name, cnts, roots, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    cnts: [num_img_pairs_in_KITTI, num_img_pairs_in_KITTI2015]
    roots: data root paths for KITTI and KITTI2015
    num_shards: integer number of shards for this data set.
  """          
  
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, sum(cnts), FLAGS.num_threads + 1).astype(np.int)
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
    args = (thread_index, ranges, name, cnts, roots, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), sum(cnts)))
  sys.stdout.flush()


def main(unused_argv):  
  train_cnts = [194, 200]
  test_cnts = [195, 200]
  
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  _process_image_files('test', test_cnts, test_roots, FLAGS.validation_shards)
  _process_image_files('train', train_cnts, train_roots, FLAGS.train_shards)

if __name__ == '__main__':
  tf.app.run()
  
  
