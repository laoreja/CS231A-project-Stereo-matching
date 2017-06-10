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
 eval_image: Prepare one image for evaluation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_height', 540,
              """Provide image_height.""")
tf.app.flags.DEFINE_integer('image_width', 960,
              """Provide image_width.""") 
tf.app.flags.DEFINE_integer('cropped_height', 256,
                            """Provide cropped_height.""")
tf.app.flags.DEFINE_integer('cropped_width', 512,
                            """Provide cropped_width.""")  
tf.app.flags.DEFINE_integer('depth', 3,
              """Provide depth.""")                                                                                    
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
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
  """Generate batches of SceneFlow images for evaluation.

  Use this function as the inputs for evaluating a network.

  Note that some (minimal) image preprocessing occurs during evaluation
  including central cropping.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    left_images, right_images: Images. 4D tensor of size [batch_size, FLAGS.cropped_height,
                cropped_width, 3].
    disparitys, masks: 3D tensor of size [batch_size, FLAGS.cropped_height, cropped_width].
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    left_images, right_images, disparitys, masks = batch_inputs(
        dataset, batch_size, train=False,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=1)
  return left_images, right_images, disparitys, masks


def distorted_inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of distorted versions of SceneFlow images.

  Use this function as the inputs for training a network.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    left_images, right_images: Images. 4D tensor of size [batch_size, FLAGS.cropped_height,
                       cropped_width, 3].
    disparitys, masks: 3D tensor of size [batch_size, FLAGS.cropped_height, cropped_width].
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
  """Prepare one image for evaluation. i.e. center cropping

  Args:
    image: 3-D float Tensor
  Returns:
    3-D float Tensor of prepared image.
  """
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
#  image = tf.image.central_crop(image, central_fraction=0.875)

  image = tf.expand_dims(image, 0)
  image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.cropped_height, FLAGS.cropped_width)
  image = tf.squeeze(image, [0])
  return image


def image_preprocessing(left_image, right_image, disparity, mask, train):
  """Decode and preprocess one image for evaluation or training.

  Args:
    left_image, right_image, mask: decoded image, dtype:tf.uint8
    disparity: decoded disparity, dtype:tf.float32
    train: boolean

  Returns:
    left_image, right_image, disparity, mask:3-D float Tensors
        all are cropped, 
          if training: random crop, 
          if testing: center crop
        images are normalized to range [-1, 1]
  """
  with tf.name_scope('image_preprocessing'):
    image_shape = tf.stack([FLAGS.image_height, FLAGS.image_width, FLAGS.depth])
    disparity_shape = tf.stack([FLAGS.image_height, FLAGS.image_width, 1])
      
    left_image = tf.reshape(left_image, image_shape)
    right_image = tf.reshape(right_image, image_shape)    
    disparity = tf.reshape(disparity, disparity_shape)
    mask = tf.reshape(mask, disparity_shape)
      
    left_image = tf.image.convert_image_dtype(left_image, dtype=tf.float32)
    right_image = tf.image.convert_image_dtype(right_image, dtype=tf.float32)
    mask = tf.cast(mask, tf.float32)

    combined = tf.concat([left_image, right_image, disparity, mask], axis=2)
    if train:
      combined_crop = tf.random_crop(combined, [FLAGS.cropped_height, FLAGS.cropped_width, tf.shape(combined)[-1]])
    else:
      combined_crop = eval_image(combined)

    left_image = combined_crop[:, :, 0:FLAGS.depth]
    right_image = combined_crop[:, :, FLAGS.depth:FLAGS.depth*2]
    disparity = combined_crop[:, :, FLAGS.depth*2:FLAGS.depth*2+1]
    mask = combined_crop[:, :, FLAGS.depth*2+1:FLAGS.depth*2+2]
    
    # Finally, rescale to [-1,1] instead of [0, 1)
    left_image = tf.subtract(left_image, 0.5)
    left_image = tf.multiply(left_image, 2.0)
    right_image = tf.subtract(right_image, 0.5)
    right_image = tf.multiply(right_image, 2.0)

    return left_image, right_image, disparity, mask


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data_SceneFlow.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

    left_image_raw: string containing encoded image in RGB colorspace
    right_image_raw: string containing encoded image in RGB colorspace
    disparity_raw: string containing float formatted grount-truth disparity
    mask_raw: string containing unit8 formatted grount-truth disparity mask

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    left_image, right_image, mask: decoded image, dtype:tf.uint8
    disparity: decoded disparity, dtype:tf.float32
    
  """
  # Dense features in Example proto.
  feature_map = {
                'left_image_raw':tf.FixedLenFeature([], tf.string),
                'right_image_raw':tf.FixedLenFeature([], tf.string),
                'mask_raw': tf.FixedLenFeature([], tf.string),
                'disparity_raw': tf.FixedLenFeature([], tf.string),
                'filename': tf.FixedLenFeature([], tf.string),
                'relative_dir': tf.FixedLenFeature([], tf.string), 
                }

  features = tf.parse_single_example(example_serialized, feature_map)
  
  left_image = tf.decode_raw(features['left_image_raw'], tf.uint8)
  right_image = tf.decode_raw(features['right_image_raw'], tf.uint8)
  mask = tf.decode_raw(features['mask_raw'], tf.uint8)
  disparity = tf.decode_raw(features['disparity_raw'], tf.float32)
    
  return left_image, right_image, disparity, mask


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
                                                      shuffle=True,
                                                      capacity=16)
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
    train_examples_per_shard = 35
    eval_examples_per_shard = 303
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
          capacity=eval_examples_per_shard + 3 * batch_size,
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
      left_image, right_image, disparity, mask = parse_example_proto(
          example_serialized)
      left_image, right_image, disparity, mask = image_preprocessing(left_image, right_image, disparity, mask, train)
      data.append([left_image, right_image, disparity, mask])

    left_images, right_images, disparitys, masks = tf.train.batch_join(
        data,
        batch_size=batch_size,
        capacity=2 * num_preprocess_threads * batch_size,
        shapes=[[FLAGS.cropped_height, FLAGS.cropped_width, FLAGS.depth], 
                [FLAGS.cropped_height, FLAGS.cropped_width, FLAGS.depth], 
                [FLAGS.cropped_height, FLAGS.cropped_width, 1],
                [FLAGS.cropped_height, FLAGS.cropped_width, 1]])
        
#    shape_op = tf.shape(left_images)

    # Display the training images in the visualizer.
#    tf.summary.image('left_images', left_images)
#    tf.summary.image('right_images', right_images)
#    tf.summary.image('disparity_with_mask', disparitys * masks)

    disparitys = tf.squeeze(disparitys, axis=3)
    masks = tf.squeeze(masks, axis=3)
    return left_images, right_images, disparitys, masks
