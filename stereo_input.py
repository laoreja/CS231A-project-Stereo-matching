# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""CIFAR dataset input module.
"""

import tensorflow as tf



#def build_input(dataset, data_path, batch_size, mode):
def build_input(dataset, batch_size, mode):  
  """Build CIFAR image and labels.

  Args:
    dataset: Either 'cifar10' or 'cifar100'.
    data_path: Filename for data.
    batch_size: Input batch size.
    mode: Either 'train' or 'eval'.
  Returns:
    images: Batches of images. [batch_size, image_size, image_size, 3]
    labels: Batches of labels. [batch_size, num_classes]
  Raises:
    ValueError: when the specified dataset is not supported.
  """
  if dataset == 'SceneFlow':
    image_height = 540
    image_width = 960
    
    cropped_height = 256
    cropped_width = 512
    
    depth = 3
  else:
    raise ValueError('Not supported dataset %s', dataset)
    
  
  with tf.variable_scope('build_input', reuse=False):
    with tf.variable_scope('queue_reader', reuse=False):
      # For debug only, really small dataset
      tfrecords_filename1 = '/home/laoreja/dataset/SceneFlow/tfrecords/train-00000-of-00012'
      tfrecords_filename2 = '/home/laoreja/dataset/SceneFlow/tfrecords/train-00001-of-00012'
      filename_queue = tf.train.string_input_producer([tfrecords_filename1, tfrecords_filename2])
        
      # Read examples from files in the filename queue.
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
    
      # Convert these examples to dense labels and processed images.
      features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
              'left_image_raw':tf.FixedLenFeature([], tf.string),
              'right_image_raw':tf.FixedLenFeature([], tf.string),
              'mask_raw': tf.FixedLenFeature([], tf.string),
              'disparity_raw': tf.FixedLenFeature([], tf.string),
              'filename': tf.FixedLenFeature([], tf.string),
              'relative_dir': tf.FixedLenFeature([], tf.string), 
              })
    
    with tf.variable_scope('process', reuse=False):
      left_image = tf.decode_raw(features['left_image_raw'], tf.uint8)
      right_image = tf.decode_raw(features['right_image_raw'], tf.uint8)
      mask = tf.decode_raw(features['mask_raw'], tf.uint8)
      disparity = tf.decode_raw(features['disparity_raw'], tf.float32)
        
      image_shape = tf.stack([image_height, image_width, depth])
      disparity_shape = tf.stack([image_height, image_width, 1])
    
      left_image = tf.reshape(left_image, image_shape)
      right_image = tf.reshape(right_image, image_shape)    
      disparity = tf.reshape(disparity, disparity_shape)
      mask = tf.reshape(mask, disparity_shape)
    
      left_image = tf.cast(left_image, tf.float32)
      right_image = tf.cast(right_image, tf.float32)
      mask = tf.cast(mask, tf.float32)

    #  if mode == 'train':
      combined = tf.concat([left_image, right_image, disparity, mask], axis=2) # MASK
      combined_crop = tf.random_crop(combined, [cropped_height, cropped_width, tf.shape(combined)[-1]])
      left_image = combined_crop[:, :, 0:depth]
      right_image = combined_crop[:, :, depth:depth*2]
      disparity = combined_crop[:, :, depth*2:depth*2+1]
      mask = combined_crop[:, :, depth*2+1:depth*2+2]
      
    #    image = tf.image.random_flip_left_right(image)
      # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
      # image = tf.image.random_brightness(image, max_delta=63. / 255.)
      # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
      left_image = tf.image.per_image_standardization(left_image)
      right_image = tf.image.per_image_standardization(right_image)
    
    with tf.variable_scope('build_queue', reuse=False):
      example_queue = tf.RandomShuffleQueue(
          capacity=64 * batch_size,
          min_after_dequeue=0 * batch_size,
          dtypes=[tf.float32, tf.float32, tf.float32, tf.float32],
          shapes=[[cropped_height, cropped_width, depth], 
                  [cropped_height, cropped_width, depth], 
                  [cropped_height, cropped_width, 1],
                  [cropped_height, cropped_width, 1]])
      num_threads = 16
  #  else:
  #    raise ValueError('Not implemented mode')
  #    image = tf.image.resize_image_with_crop_or_pad(
  #        image, image_size, image_size)
  #    image = tf.image.per_image_standardization(image)
  #
#    with tf.variable_scope('build_queue', reuse=False):
#      example_queue = tf.FIFOQueue(
#          1 * batch_size,
#          dtypes=[tf.float32, tf.float32, tf.float32, tf.float32],
#          shapes=[[cropped_height, cropped_width, cropped_depth], [cropped_height, cropped_width, cropped_depth], [cropped_height, cropped_width, 1], [cropped_height, cropped_width, 1]])
#      num_threads = 1

    with tf.variable_scope('enqueue', reuse=False):
      example_enqueue_op = example_queue.enqueue([left_image, right_image, disparity, mask]) 
      tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
          example_queue, [example_enqueue_op] * num_threads))

    with tf.variable_scope('dequeue', reuse=False):
      # Read 'batch' labels + images from the example queue.
      left_images, right_images, disparitys, masks = example_queue.dequeue_many(batch_size) # masks
      
      tf.summary.image('true_disparity', disparity*masks)
      
      disparitys = tf.squeeze(disparitys, axis=3)
      masks = tf.squeeze(masks, axis=3)
    
#      assert len(left_images.get_shape()) == 4
#      assert left_images.get_shape()[0] == batch_size
#      assert left_images.get_shape()[-1] == 3
#      assert len(disparitys.get_shape()) == 3
#      assert disparitys.get_shape()[0] == batch_size

      # Display the training images in the visualizer.
      tf.summary.image('left_images', left_images)
      tf.summary.image('right_images', right_images)  
    return left_images, right_images, disparitys, masks
