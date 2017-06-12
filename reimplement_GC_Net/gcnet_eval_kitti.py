# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
Evaluation for GC-Net on KITTI.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf

import gcnet_model
import image_processing_KITTI
from KITTI_data import KITTIData
#from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 1
#NUM_EVAL_SAMPLES = 4843

#tf.app.flags.DEFINE_string('dataset', 'KITTI', '')
tf.app.flags.DEFINE_string('mode', 'eval', 'mode')
tf.app.flags.DEFINE_boolean('debug', False,
              """Whether to show verbose summaries.""")
tf.app.flags.DEFINE_string('log_root', '/home/laoreja/tf/log/kitti_from_retrain_6',
                           """Directory where to write event logs' parent.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/laoreja/tf/log/kitti_from_retrain_6/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, op_list, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    op_list: List of ops for evaluation. Need computing the average of each of them.
    summary_op: Summary op.
  """
  len_op_list = len(op_list)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      tf.logging.info('restore from %s' % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      tf.logging.info('starts evaluation!')

      num_iter = NUM_EVAL_SAMPLES
      avg_list = [0.0 for i in range(len_op_list)]
      step = 0
      while step < num_iter and not coord.should_stop():
        res_list = sess.run(op_list)
        for i in range(len_op_list):
          avg_list[i] += res_list[i]
        step += 1
        print(res_list)
        
        if step % 50 == 0:
#          print('step: ', step, res_list)
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

      # Compute averages of evaluating criteria
      for i in range(len_op_list):
        avg_list[i] /= (num_iter * 1.0)
      
      print('avg res:', avg_list)

#      summary = tf.Summary()
#      summary.ParseFromString(sess.run(summary_op))
      
#      for i in range(len_op_list):
#        summary.value.add(tag='avg_'+str(i), simple_value=avg_list[i])
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(hps, dataset):
  """Eval for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get inputs. You may write them to summary.
    num_preprocess_threads = FLAGS.num_preprocess_threads
    left_images, right_images, disparitys, masks = image_processing_KITTI.inputs(
                dataset,
                batch_size = hps.batch_size,
                num_preprocess_threads=num_preprocess_threads)
    tf.summary.image('left_image', left_images)
    tmp = tf.expand_dims(disparitys, axis=3)
    tf.summary.image('disparity', tmp)
    tmp = tf.expand_dims(masks, axis=3)
    tf.summary.image('masks', tmp)
    tf.summary.image('masked_disparity', tf.expand_dims(disparitys * masks, axis=3))

    # Build a Graph that computes the disparity predictions from the
    # inference model.
    model = gcnet_model.GCNet(hps, left_images, right_images, disparitys, masks, 'eval') # 
    model.build_graph_to_loss()

    # Restore the moving average version of the learned variables for eval.
#    variable_averages = tf.train.ExponentialMovingAverage(
#        gcnet_model.MOVING_AVERAGE_DECAY)
#    variables_to_restore = variable_averages.variables_to_restore()
#    saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    tf.summary.image('predict', tf.expand_dims(model.predicted_disparity, axis=3))        
    tf.summary.image('masked_predict', tf.expand_dims(model.predicted_disparity * masks, axis=3))
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, [model.abs_loss, model.larger_than_3px, model.larger_than_5px, model.larger_than_7px], summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

  dataset = KITTIData('train')
  assert dataset.data_files()
  global NUM_EVAL_SAMPLES
  NUM_EVAL_SAMPLES = dataset.num_examples_per_epoch()
  
  FLAGS.eval_dir = os.path.join(FLAGS.log_root, 'train_eval')
  
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)

# Indeed, the lrn_rate and weight_decay_rate have no use.
  hps = gcnet_model.HParams(batch_size=BATCH_SIZE,
                             lrn_rate=0.0,
                             weight_decay_rate=0.0,
                             relu_leakiness=0.1,
                             optimizer='RMSProp',
                             max_disparity=192) 

  evaluate(hps, dataset)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)    
  tf.app.run()
