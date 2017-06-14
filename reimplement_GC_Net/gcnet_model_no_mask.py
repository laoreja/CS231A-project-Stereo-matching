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

"""
GC-Net model.

https://arxiv.org/pdf/1703.04309.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages

# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


HParams = namedtuple('HParams',
                     ['batch_size', 'lrn_rate',
                     'weight_decay_rate',
                     'relu_leakiness', 'optimizer', 'max_disparity'])


class GCNet(object):
  """GC-Net model."""

  def __init__(self, hps, left_images, right_images, gt_disparity, mode): 
    """GC-Net constructor.

    Args:
      hps: Hyperparameters.
      left_images, right_images: Batches of images. [batch_size, image_height, image_width, 3]
      gt_disparity: Batches of disparity. [batch_size, image_height, image_width]
      mode: One of 'train', 'eval' and 'predict'.
    """
    self.hps = hps
    self._left_images = left_images
    self._right_images = right_images
    self.gt_disparity = gt_disparity
    self.mode = mode
    self.debug_op_list = []  

    self._extra_train_ops = []
    
  def build_graph_to_loss(self):
    self._build_model()
    self._build_loss_op()
#    self._add_loss_summaries()
    self.variables_to_restore = tf.get_collection('variable_to_restore')
    self.summaries = tf.summary.merge_all()

  def build_graph(self):
    """Build a whole graph for the model."""
#    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self._build_model()
    if self.mode != 'predict':
      self._build_loss_op()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.summary.merge_all()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]
    
  def _stride_3d_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv3d."""
    return [1, stride, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""

    layer_idx = 1
    with tf.variable_scope('unary_features', reuse=False):
      with tf.variable_scope('layer_'+str(layer_idx)):
        layer_idx += 1
        left_x = self._left_images
        left_x = self._conv('conv', left_x, 5, 3, 32, self._stride_arr(2))
        left_x = self._relu(left_x, self.hps.relu_leakiness)
        left_x = self._batch_norm('bn', left_x)
#      tf.add_to_collection('shapes', tf.shape(left_x))
        
      for i in six.moves.range(8):
        left_x, layer_idx = self._unary_feat_residual(left_x, 3, 32, 32, self._stride_arr(1), layer_idx)
#        tf.add_to_collection('shapes', tf.shape(left_x))
    
      with tf.variable_scope('layer_'+str(layer_idx)):
        layer_idx += 1
        left_x = self._conv('conv', left_x, 3, 32, 32, self._stride_arr(1))
      #tf.add_to_collection('shapes', tf.shape(left_x))
    
    layer_idx = 1    
    with tf.variable_scope('unary_features', reuse=True):
      with tf.variable_scope('layer_'+str(layer_idx)):
        layer_idx += 1
        right_x = self._left_images
        right_x = self._conv('conv', right_x, 5, 3, 32, self._stride_arr(2))
        right_x = self._relu(right_x, self.hps.relu_leakiness)
        right_x = self._batch_norm('bn', right_x)
        
      for i in six.moves.range(8):
        right_x, layer_idx = self._unary_feat_residual(right_x, 3, 32, 32, self._stride_arr(1), layer_idx)

      with tf.variable_scope('layer_'+str(layer_idx)):
        layer_idx += 1
        right_x = self._conv('conv', right_x, 3, 32, 32, self._stride_arr(1))
      
#      self.left_x_shape_op = tf.shape(left_x)
#      self.right_x_shape_op = tf.shape(right_x)

    with tf.variable_scope('cost_volumn'):
      left_cost_volume = tf.stack([tf.identity(left_x)] * (self.hps.max_disparity/2+1), axis=1, name='left_stack')
      right_cost_volume = []
      cur_width = tf.shape(right_x)[2]

      for depth in six.moves.range(self.hps.max_disparity/2+1):
        right_cost_volume.append(tf.pad(tf.slice(right_x, [0, 0, 0, 0], [-1, -1, cur_width - depth, -1], name='right_slice_'+str(depth)),
                                        [[0, 0], [0, 0], [depth, 0], [0, 0]],
                                        name='right_pad_'+str(depth)
                                        ))
      right_cost_volume = tf.stack(right_cost_volume, axis=1, name='right_stack')
      x = tf.concat([left_cost_volume, right_cost_volume], 4)
      #tf.add_to_collection('shapes', tf.shape(x))
      
          
    with tf.variable_scope('learning_regularization'):
      stored_features = []

      in_filters = [64, 64, 64, 64]
      out_filters = [32, 64, 64, 64]
      in_filters_stride_2 = [64, 64, 64, 64]
      out_filters_stride_2 = [64, 64, 64, 128]
      for i in six.moves.range(4):
        tmp_x, layer_idx = self._regularization_subsample(x, 3, in_filters[i], out_filters[i], self._stride_3d_arr(1), layer_idx)
#        tf.add_to_collection('shapes', tf.shape(tmp_x))
        stored_features.append(tmp_x)
#        stored_features[i] = tmp_x
#        self._extra_train_ops.append(stored_features[i].assign(tmp_x)) ## it's an op, how to add it to the graph?
        
        with tf.variable_scope('layer_'+str(layer_idx)):
          layer_idx += 1
          x = self._conv3d('conv3d', x, 3, in_filters_stride_2[i], out_filters_stride_2[i], self._stride_3d_arr(2))
          x = self._relu(x, self.hps.relu_leakiness)
          x = self._batch_norm('bn', x)
#          tf.add_to_collection('shapes', tf.shape(x))

      
      assert stored_features[0] is not stored_features[1]

      for i in six.moves.range(2):
        with tf.variable_scope('layer_'+str(layer_idx)):
          layer_idx += 1
          x = self._conv3d('conv3d', x, 3, 128, 128, self._stride_3d_arr(1))
          x = self._relu(x, self.hps.relu_leakiness)
          x = self._batch_norm('bn', x)
#          tf.add_to_collection('shapes', tf.shape(x))

      transposed_in_filters = [128, 64, 64, 64]
      transposed_out_filters = [64, 64, 64, 32]
      
      for i in six.moves.range(4):
        x, layer_idx = self._regularization_upsample(x, stored_features[-i-1], 3, transposed_in_filters[i], transposed_out_filters[i], self._stride_3d_arr(2), layer_idx)
#        tf.add_to_collection('shapes', tf.shape(x))
      
      with tf.variable_scope('layer_'+str(layer_idx)):
        layer_idx += 1
        input_shape = tf.shape(self.gt_disparity)
        x = self._conv3d_trans('conv_trans', x, 3, 32, 1, self._stride_3d_arr(2), [input_shape[0], self.hps.max_disparity+1, input_shape[1], input_shape[2], 1])
#        tf.add_to_collection('shapes', tf.shape(x))

    
    with tf.variable_scope('soft_argmin'):
        x = tf.squeeze(x, squeeze_dims=[4], name='squeeze')
#        tf.add_to_collection('shapes', tf.shape(x))
        x = tf.transpose(x, perm=[0, 2, 3, 1], name='transpose')
#        tf.add_to_collection('shapes', tf.shape(x))
        x = tf.nn.softmax(x, dim=-1, name='softmax')
#        tf.add_to_collection('shapes', tf.shape(x))

        multiplier = tf.range(0, self.hps.max_disparity+1, dtype=tf.float32, name='depth_range')
        x = tf.multiply(x, multiplier, name='softmax_mul_depth')
#        tf.add_to_collection('shapes', tf.shape(x))
        self.predicted_disparity = tf.reduce_sum(x, axis=3, name='reduce_sum') 
        
#        tf.summary.image('left', self._left_images)
#        tf.summary.image('gt', tf.expand_dims(self.gt_disparity, axis=3))                    
#        tf.summary.image('predict', tf.expand_dims(self.predicted_disparity, axis=3))        

#        tf.add_to_collection('shapes', tf.shape(self.predicted_disparity))
#    self.shapes = tf.get_collection('shapes')
#    self.debug_op_list.append(self.shapes)

  def _build_loss_op(self):
    with tf.variable_scope('loss'):
      self.abs_loss = tf.reduce_mean(tf.abs(self.gt_disparity - self.predicted_disparity), name='abs_loss')
#      tf.summary.scalar('abs_loss', self.abs_loss)
      self.total_loss = self.abs_loss + self._decay()
#      tf.summary.scalar('total_loss', self.total_loss)
      tf.summary.image('left', self._left_images)
      #        tf.summary.image('gt', tf.expand_dims(self.gt_disparity, axis=3))                    
      tf.summary.image('predict', tf.expand_dims(self.predicted_disparity, axis=3))  
      if self.mode == 'eval':
        self.larger_than_3px = tf.reduce_mean(tf.cast(tf.greater(tf.abs(self.gt_disparity - self.predicted_disparity), 3), tf.float32))
        self.larger_than_5px = tf.reduce_mean(tf.cast(tf.greater(tf.abs(self.gt_disparity - self.predicted_disparity), 5), tf.float32))
        self.larger_than_7px = tf.reduce_mean(tf.cast(tf.greater(tf.abs(self.gt_disparity - self.predicted_disparity), 7), tf.float32))
        
#      tf.summary.histogram('loss_gradints', tf.gradients(self.total_loss, self.predicted_disparity)[0])
      
  def _add_loss_summaries(self):
    """Add summaries for losses in GC-Net model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='loss_avg')
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
      self.loss_averages_op = loss_averages.apply([self.abs_loss, self.total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in [self.abs_loss, self.total_loss]:
      # Name each loss as '(raw)' and name the moving average version of the loss
      # as the original loss name.
      tf.summary.scalar(l.op.name + '_raw', l)
      tf.summary.scalar(l.op.name, loss_averages.average(l))

#    self.avg_abs_loss = loss_averages.average(self.abs_loss)
#    self.avg_total_loss = loss_averages.average(self.total_loss)
#    return loss_averages_op
    
  def _build_train_op(self, global_step):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning_rate', self.lrn_rate)

    loss_averages_op = self._add_loss_summaries()
    
    with tf.control_dependencies([loss_averages_op]):
      if self.hps.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
      elif self.hps.optimizer == 'mom':
        optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
      elif self.hps.optimizer == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(self.lrn_rate, decay=0.9, momentum=0.9, epsilon=1)
        
        trainable_variables = tf.trainable_variables()
#        grads = tf.gradients(self.total_loss, trainable_variables)
        grads = optimizer.compute_gradients(self.total_loss, trainable_variables)


    apply_op = optimizer.apply_gradients(
#        zip(grads, trainable_variables),
        grads,
        global_step=global_step, 
        name='train_step')
        
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_op, variables_averages_op]):
      self.train_op = tf.no_op(name='train')
#    train_ops = [apply_op] + self._extra_train_ops
#    self.train_op = tf.group(*train_ops)


  def _regularization_upsample(self, x, feature, filter_size, in_filter, out_filter, stride, layer_idx):
    with tf.variable_scope('layer_'+str(layer_idx)):
      layer_idx += 1
      x = self._conv3d_trans('conv_trans', x, filter_size, in_filter, out_filter, stride, tf.shape(feature))
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._batch_norm('bn', x)
      
    with tf.variable_scope('residual_after_'+str(layer_idx-1)):
      x += feature

    tf.logging.debug('image after unit %s', x.get_shape())
    return x, layer_idx

  def _regularization_subsample(self, x, filter_size, in_filter, out_filter, stride, layer_idx):

    with tf.variable_scope('layer_'+str(layer_idx)):
      layer_idx += 1
      x = self._conv3d('conv3d', x, filter_size, in_filter, out_filter, stride)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._batch_norm('bn', x)

    with tf.variable_scope('layer_'+str(layer_idx)):
      layer_idx += 1
      x = self._conv3d('conv3d', x, filter_size, out_filter, out_filter, stride)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._batch_norm('bn', x)
      
    tf.logging.debug('image after unit %s', x.get_shape())
    return x, layer_idx

  def _unary_feat_residual(self, x, filter_size, in_filter, out_filter, stride, layer_idx):
    orig_x = x
    orig_layer_idx = layer_idx - 1
    
    for i in six.moves.range(2):
      with tf.variable_scope('layer_'+str(layer_idx)):
        layer_idx += 1
        x = self._conv('conv', x, 3, in_filter, out_filter, stride)
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._batch_norm('bn', x)
          
    with tf.variable_scope('residual_btw_'+str(layer_idx-1)+'_'+str(orig_layer_idx)):
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x, layer_idx


  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)

    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = self._variable_on_cpu(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')
      
  def _conv3d(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution 3D."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * filter_size * out_filters
      kernel = self._variable_on_cpu(
          'DW', [filter_size, filter_size, filter_size, in_filters, out_filters],
           initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv3d(x, kernel, strides, padding='SAME')
      
  def _conv3d_trans(self, name, x, filter_size, in_filters, out_filters, strides, output_shape):
    """Convolution 3D transpose."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * filter_size * out_filters
      kernel = self._variable_on_cpu(
          'DW', [filter_size, filter_size, filter_size, out_filters, in_filters],
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv3d_transpose(
                x, 
                kernel, 
                output_shape,
                strides, 
                padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = self._variable_on_cpu(
          'beta', params_shape,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = self._variable_on_cpu(
          'gamma', params_shape,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
#        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
        mean, variance = tf.nn.moments(x, range(len(x.get_shape())-1), name='moments')

        moving_mean = self._variable_on_cpu(
            'moving_mean', params_shape,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = self._variable_on_cpu(
            'moving_variance', params_shape,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, BATCHNORM_MOVING_AVERAGE_DECAY))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, BATCHNORM_MOVING_AVERAGE_DECAY))
      else:
        mean, variance = tf.nn.moments(x, range(len(x.get_shape())-1), name='moments')
#        mean = self._variable_on_cpu(
#            'moving_mean', params_shape,
#            initializer=tf.constant_initializer(0.0, tf.float32),
#            trainable=False)
#        variance = self._variable_on_cpu(
#            'moving_variance', params_shape,
#            initializer=tf.constant_initializer(1.0, tf.float32),
#            trainable=False)
#        tf.summary.histogram(mean.op.name, mean)
#        tf.summary.histogram(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _variable_on_cpu(self, name, shape, initializer, dtype=tf.float32, trainable=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
      tf.add_to_collection("variable_to_restore", var)
    return var

#  def _fully_connected(self, x, out_dim):
#    """FullyConnected layer for final output."""
#    x = tf.reshape(x, [self.hps.batch_size, -1])
#    w = tf.get_variable(
#        'DW', [x.get_shape()[1], out_dim],
#        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
#    b = tf.get_variable('biases', [out_dim],
#                        initializer=tf.constant_initializer())
#    return tf.nn.xw_plus_b(x, w, b)
#
#  def _global_avg_pool(self, x):
#    assert x.get_shape().ndims == 4
#    return tf.reduce_mean(x, [1, 2])
