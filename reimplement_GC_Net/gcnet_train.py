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
GC-Net Train module.
Using single GPU/CPU.
"""
import time
import six
import sys, os

import stereo_input
import numpy as np
import tensorflow as tf

import gcnet_model
import image_processing
from SceneFlow_data import SceneFlowData

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'SceneFlow', '')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('max_steps', 150000, 'max steps for training')                            
tf.app.flags.DEFINE_integer('batch_size', 1,
              'batch_size')


def train(hps, dataset):
  """Training loop."""
  num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
  left_images, right_images, disparitys, masks = image_processing.distorted_inputs(
    dataset,
    batch_size = hps.batch_size
    num_preprocess_threads=num_preprocess_threads)
  model = gcnet_model.GCNet(hps, left_images, right_images, disparitys, masks, FLAGS.mode)
  model.build_graph()
  
  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)  
  
  
  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=model.summaries)
#      tf.summary.merge([model.summaries,
#                                    tf.summary.scalar('average_abs_loss', average_abs_loss),
#                                    tf.summary.scalar('average_total_loss', average_total_loss)
#                                  ]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
                'total_loss': model.total_loss,
#                'average_abs_loss': average_abs_loss,
#                'average_total_loss':average_total_loss
                },
      every_n_iter=100)

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:        
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  batch_size = 1

  dataset = SceneFlowData('train')
  assert dataset.data_files()
  
  if tf.gfile.Exists(FLAGS.root_dir):
    tf.gfile.DeleteRecursively(FLAGS.root_dir)
  tf.gfile.MakeDirs(FLAGS.root_dir)

  FLAGS.train_dir = os.path.join(FLAGS.log_root, 'train')

  hps = gcnet_model.HParams(batch_size=batch_size,
                             lrn_rate=0.01,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='RMSProp',
                             max_disparity=200) 

  with tf.device(dev):
    train(hps, dataset)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
