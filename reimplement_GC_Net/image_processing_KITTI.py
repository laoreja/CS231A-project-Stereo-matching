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
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
  of an image.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 eval_image: resize images
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('cropped_height', 388,
                            """Provide cropped_height.""")
tf.app.flags.DEFINE_integer('cropped_width', 1240,
                            """Provide cropped_width.""")  
tf.app.flags.DEFINE_integer('half_height', 194,
              """Provide cropped_height.""")
tf.app.flags.DEFINE_integer('half_width', 620,
                        """Provide cropped_width.""")                            
tf.app.flags.DEFINE_integer('depth', 3,
              """Provide depth.""")                                                                                    
tf.app.flags.DEFINE_integer('num_preprocess_threads', 1,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")

print("import image processing")

def inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of KITTI images for evaluation.

  Use this function as the inputs for evaluating a network.

  Note that some (minimal) image preprocessing occurs during evaluation
  including central cropping.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    left_images, right_images: Images. 4D tensor of size [batch_size, FLAGS.half_height, FLAGS.half_width, 3].
    disparitys, masks: 3D tensor of size [batch_size, FLAGS.half_height, FLAGS.half_width].
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    left_images, right_images, filenames, subsets = batch_inputs(
        dataset, batch_size, train=False,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=1)
  return left_images, right_images, filenames, subsets


def distorted_inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of distorted versions of KITTI images.

  Use this function as the inputs for training a network.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    left_images, right_images: Images. 4D tensor of size [batch_size, FLAGS.half_height, FLAGS.half_width, 3].
    disparitys, masks: 3D tensor of size [batch_size, FLAGS.half_height, FLAGS.half_width].
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    left_images, right_images, disparitys, masks = batch_inputs(
        dataset, batch_size, train=True,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=FLAGS.num_readers)
  return left_images, right_images, disparitys, masks



def eval_image(image):
  """Resize images

  Args:
    image: 3-D float Tensor
  Returns:
    3-D float Tensor of prepared image.
  """
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
#  image = tf.image.central_crop(image, central_fraction=0.875)

  # Resize the image to the original height and width.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.cropped_height, FLAGS.cropped_width)
  image = tf.squeeze(image, [0])
  return image


def image_preprocessing(left_image, right_image, disparity, mask, height, width):
  """Decode and preprocess one image for evaluation or training.

  Args:
    left_image, right_image, mask: decoded image, dtype:tf.uint8
    disparity: decoded disparity, dtype:tf.float32
    train: boolean

  Returns:
    left_image, right_image, disparity, mask:3-D float Tensors
        Since the H and W are too large, I divide the images into 4 parts, each with size H/2 * W/2
        images are normalized to range [-1, 1]
  """
  with tf.name_scope('image_preprocessing'):
    image_shape = tf.stack([height, width, FLAGS.depth])
    left_image = tf.reshape(left_image, image_shape)
    right_image = tf.reshape(right_image, image_shape) 

    left_image = tf.image.convert_image_dtype(left_image, dtype=tf.float32)
    right_image = tf.image.convert_image_dtype(right_image, dtype=tf.float32)

    if disparity is not None:
      disparity_shape = tf.stack([height, width, 1])
      disparity = tf.reshape(disparity, disparity_shape)
      mask = tf.reshape(mask, disparity_shape)
      mask = tf.cast(mask, tf.float32)

      combined = tf.concat([left_image, right_image, disparity, mask], axis=2)
    else:
      combined = tf.concat([left_image, right_image], axis=2)
      
    combined_crop = eval_image(combined)
    
    left_image = combined_crop[:, :, 0:FLAGS.depth]
    right_image = combined_crop[:, :, FLAGS.depth:FLAGS.depth*2]
    # Finally, rescale to [-1,1] instead of [0, 1)
    left_image = tf.subtract(left_image, 0.5)
    left_image = tf.multiply(left_image, 2.0)
    right_image = tf.subtract(right_image, 0.5)
    right_image = tf.multiply(right_image, 2.0)
#    tf.summary.image('origin_left', tf.expand_dims(left_image, 0))
#    tf.summary.image('origin_right', tf.expand_dims(right_image, 0))
    
    left_images = [left_image[0:FLAGS.half_height, 0:FLAGS.half_width], 
                   left_image[0:FLAGS.half_height, FLAGS.half_width:],
                   left_image[FLAGS.half_height:, FLAGS.half_width:],
                   left_image[FLAGS.half_height:, 0:FLAGS.half_width]]
    right_images = [right_image[0:FLAGS.half_height, 0:FLAGS.half_width], 
             right_image[0:FLAGS.half_height, FLAGS.half_width:],
             right_image[FLAGS.half_height:, FLAGS.half_width:],
             right_image[FLAGS.half_height:, 0:FLAGS.half_width]]
#    tf.summary.image('new_left', left_images)
#    tf.summary.image('new_right', right_images)    
    
    if disparity is not None:
      disparity = combined_crop[:, :, FLAGS.depth*2:FLAGS.depth*2+1]
      mask = combined_crop[:, :, FLAGS.depth*2+1:FLAGS.depth*2+2]      

      disparitys = [disparity[0:FLAGS.half_height, 0:FLAGS.half_width], 
               disparity[0:FLAGS.half_height, FLAGS.half_width:],
               disparity[FLAGS.half_height:, FLAGS.half_width:],
               disparity[FLAGS.half_height:, 0:FLAGS.half_width]]
      masks = [mask[0:FLAGS.half_height, 0:FLAGS.half_width], 
            mask[0:FLAGS.half_height, FLAGS.half_width:],
            mask[FLAGS.half_height:, FLAGS.half_width:],
            mask[FLAGS.half_height:, 0:FLAGS.half_width]]              
#      tf.summary.image('origin_disparity', tf.expand_dims(disparity, 0))
#      tf.summary.image('new_disparity', disparitys)    

  
      return left_images, right_images, disparitys, masks
    return left_images, right_images


def parse_example_proto(example_serialized, train):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data_KITTI.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

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


  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    left_image, right_image, mask: decoded image, dtype:tf.uint8
    disparity: decoded disparity, dtype:tf.float32
  """
  # Dense features in Example proto.
  if train:
    feature_map = {
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'left_image_raw':tf.FixedLenFeature([], tf.string),
      'right_image_raw':tf.FixedLenFeature([], tf.string),
      'mask_raw': tf.FixedLenFeature([], tf.string),
      'disparity_raw': tf.FixedLenFeature([], tf.string),
      'filename': tf.FixedLenFeature([], tf.string),
      'subset_idx':tf.FixedLenFeature([], tf.int64)
      }
  else:
    feature_map = {
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'left_image_raw':tf.FixedLenFeature([], tf.string),
      'right_image_raw':tf.FixedLenFeature([], tf.string),
      'filename': tf.FixedLenFeature([], tf.string),
      'subset_idx':tf.FixedLenFeature([], tf.int64)
      }

  features = tf.parse_single_example(example_serialized, feature_map)
  
  left_image = tf.decode_raw(features['left_image_raw'], tf.uint8)
  right_image = tf.decode_raw(features['right_image_raw'], tf.uint8)

  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  subset_idx = tf.cast(features['subset_idx'], tf.int32)
  
  if train:
    disparity = tf.decode_raw(features['disparity_raw'], tf.float32)
    mask = tf.decode_raw(features['mask_raw'], tf.uint8)
    return left_image, right_image, disparity, mask, height, width
  
  return left_image, right_image, height, width, features['filename'], subset_idx


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
                 num_readers=1):
  """Contruct batches of training or evaluation examples from the image dataset.

  Args:
    dataset: instance of Dataset class specifying the dataset.
      See dataset.py for details.
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers

  Returns:
    left_images, right_images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                image_size, 3].
    disparitys, masks: 3D tensor of size [batch_size, FLAGS.image_size, image_size].

  Raises:
    ValueError: if data is not found
  """
  with tf.name_scope('batch_processing'):
    data_files = dataset.data_files()
    if data_files is None:
      raise ValueError('No data files found for this dataset')

    # Create filename_queue
    if train:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)
    else:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)
    if num_preprocess_threads is None:
      num_preprocess_threads = FLAGS.num_preprocess_threads

    if num_preprocess_threads % 4:
      raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
      num_readers = FLAGS.num_readers

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    train_examples_per_shard = 50
    test_examples_per_shard = 50
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    min_queue_examples = train_examples_per_shard * FLAGS.input_queue_memory_factor
    if train:
      examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])
    else:
      examples_queue = tf.FIFOQueue(
          capacity=test_examples_per_shard + 3 * batch_size,
          dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
        reader = dataset.reader()
        _, value = reader.read(filename_queue)
        enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
    else:
      reader = dataset.reader()
      _, example_serialized = reader.read(filename_queue)

    data = []
    for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      if train:
        left_image, right_image, disparity, mask, height, width = parse_example_proto(
          example_serialized, train)
        left_images, right_images, disparitys, masks = image_preprocessing(left_image, right_image, disparity, mask, height, width)
        data.extend([list(item) for item in zip(left_images, right_images, disparitys, masks)])

        left_images, right_images, disparitys, masks = tf.train.batch_join(
            data,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size,
            shapes=[[FLAGS.half_height, FLAGS.half_width, FLAGS.depth], 
                    [FLAGS.half_height, FLAGS.half_width, FLAGS.depth], 
                    [FLAGS.half_height, FLAGS.half_width, 1],
                    [FLAGS.half_height, FLAGS.half_width, 1]])

#        tf.summary.image('left_images', left_images)
#        tf.summary.image('right_images', right_images)
#        tf.summary.image('disparitys', disparitys)
        
        disparitys = tf.squeeze(disparitys, axis=3)
        masks = tf.squeeze(masks, axis=3)
        return left_images, right_images, disparitys, masks
      else:
        left_image, right_image, height, width, filename, subset_idx = parse_example_proto(
          example_serialized, train)
        left_images, right_images = image_preprocessing(left_image, right_image, None, None, height, width)
        filenames = [filename] * 4
        subset_idxs = [subset_idx] * 4
        data.extend([list(item) for item in zip(left_images, right_images, filenames, subset_idxs)])

        left_images, right_images, filenames, subset_idxs = tf.train.batch_join(
          data,
          batch_size=batch_size,
          capacity=2 * num_preprocess_threads * batch_size,
          shapes=[[FLAGS.half_height, FLAGS.half_width, FLAGS.depth], 
              [FLAGS.half_height, FLAGS.half_width, FLAGS.depth],
              [],
              []])
#        tf.summary.image('left_images', left_images)
#        tf.summary.image('right_images', right_images)
        return left_images, right_images, filenames, subset_idxs
        




