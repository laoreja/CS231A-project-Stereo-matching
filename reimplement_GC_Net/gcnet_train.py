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

"""ResNet Train/Eval module.
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
#tf.app.flags.DEFINE_string('train_data_path', '',
#                           'Filepattern for training data.')
#tf.app.flags.DEFINE_string('eval_data_path', '',
#                           'Filepattern for eval data')
#tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')

#tf.app.flags.DEFINE_string('train_dir', '',
#                           'Directory to keep training outputs.')
                          
#tf.app.flags.DEFINE_string('eval_dir', '',
#                           'Directory to keep eval outputs.')
#tf.app.flags.DEFINE_integer('eval_batch_count', 50,
#                            'Number of batches to eval.')
#tf.app.flags.DEFINE_bool('eval_once', False,
#                         'Whether evaluate the model only once.')
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
  model = gcnet_model.GCNet(hps, left_images, right_images, disparitys, masks, FLAGS.mode) # 
  model.build_graph()
  
  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)  
  
#  average_abs_loss = tf.reduce_mean(model.abs_loss)
#  average_total_loss = tf.reduce_mean(model.total_loss)
#  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
#  writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
#  sess.run(tf.global_variables_initializer())
  
#  for i in range(FLAGS.max_steps):
#    print('step: ', i)
#    if i % 100 == 99:
#      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#      run_metadata = tf.RunMetadata()
#      _ = sess.run([model.train_op], options=run_options, run_metadata=run_metadata)
#      writer.add_run_metadata(run_metadata, 'step%03d' % i)
#      writer.add_summary(model.summaries, i)
#      print('Adding run metadata for', i, model.global_step)
#      writer.flush()
#    else:
#      _ = sess.run([model.train_op])
##      writer.add_summary(model.summaries, i)
##      print('write summary')
#    print('loss', model.total_loss)
#
#      
#  writer.close()
  

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


  
#  mon_sess = tf.train.MonitoredTrainingSession(
#      checkpoint_dir=FLAGS.log_root,
#      # Since we provide a SummarySaverHook, we need to disable default
#      # SummarySaverHook. To do that we set save_summaries_steps to 0.
#      save_summaries_steps=0,
#      config=tf.ConfigProto(allow_soft_placement=True))
#  while not mon_sess.should_stop():
#    print(mon_sess.run([model.predicted_disparity]))
#  mon_sess.close()
#      mon_sess.run([model.left_x_shape_op, model.right_x_shape_op, model.cost_volume_shape_op, model.train_op])

  
  
#  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#  sess.run(tf.global_variables_initializer())
#  print(sess.run([model.predicted_disparity]))
#  sess.close()

#  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
#      tf.get_default_graph(),
#      tfprof_options=tf.contrib.tfprof.model_analyzer.
#          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
#  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
#
#  tf.contrib.tfprof.model_analyzer.print_model_analysis(
#      tf.get_default_graph(),
#      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
#  
#  average_abs_loss = tf.reduce_mean(model.abs_loss)
#  average_total_loss = tf.reduce_mean(model.total_loss)
#
#  summary_hook = tf.train.SummarySaverHook(
#      save_steps=100,
#      output_dir=FLAGS.train_dir,
#      summary_op=tf.summary.merge([model.summaries,
#                                   tf.summary.scalar('average_abs_loss', average_abs_loss),
#                                   tf.summary.scalar('average_total_loss', average_total_loss)]))
#
#  logging_hook = tf.train.LoggingTensorHook(
#      tensors={'step': model.global_step,
#               'total_loss': model.total_loss,
#               'average_abs_loss': average_abs_loss,
#               'average_total_loss':average_total_loss},
#      every_n_iter=100)
#
#  with tf.train.MonitoredTrainingSession(
#      checkpoint_dir=FLAGS.log_root,
#      hooks=[logging_hook],
#      chief_only_hooks=[summary_hook],
#      # Since we provide a SummarySaverHook, we need to disable default
#      # SummarySaverHook. To do that we set save_summaries_steps to 0.
#      save_summaries_steps=0,
#      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
#    while not mon_sess.should_stop():
#      mon_sess.run([model.left_x_shape_op, model.right_x_shape_op, model.cost_volume_shape_op])
#      mon_sess.run([model.left_x_shape_op, model.right_x_shape_op, model.cost_volume_shape_op, model.train_op])


#def evaluate(hps):
#  """Eval loop."""
#  images, labels = cifar_input.build_input(
#      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
#  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
#  model.build_graph()
#  saver = tf.train.Saver()
#  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
#
#  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#  tf.train.start_queue_runners(sess)
#
#  best_precision = 0.0
#  while True:
#    try:
#      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
#    except tf.errors.OutOfRangeError as e:
#      tf.logging.error('Cannot restore checkpoint: %s', e)
#      continue
#    if not (ckpt_state and ckpt_state.model_checkpoint_path):
#      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
#      continue
#    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
#    saver.restore(sess, ckpt_state.model_checkpoint_path)
#
#    total_prediction, correct_prediction = 0, 0
#    for _ in six.moves.range(FLAGS.eval_batch_count):
#      (summaries, loss, predictions, truth, train_step) = sess.run(
#          [model.summaries, model.cost, model.predictions,
#           model.labels, model.global_step])
#
#      truth = np.argmax(truth, axis=1)
#      predictions = np.argmax(predictions, axis=1)
#      correct_prediction += np.sum(truth == predictions)
#      total_prediction += predictions.shape[0]
#
#    precision = 1.0 * correct_prediction / total_prediction
#    best_precision = max(precision, best_precision)
#
#    precision_summ = tf.Summary()
#    precision_summ.value.add(
#        tag='Precision', simple_value=precision)
#    summary_writer.add_summary(precision_summ, train_step)
#    best_precision_summ = tf.Summary()
#    best_precision_summ.value.add(
#        tag='Best Precision', simple_value=best_precision)
#    summary_writer.add_summary(best_precision_summ, train_step)
#    summary_writer.add_summary(summaries, train_step)
#    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
#                    (loss, precision, best_precision))
#    summary_writer.flush()
#
#    if FLAGS.eval_once:
#      break
#
#    time.sleep(60)


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  batch_size = 1

  if FLAGS.mode == 'train':
    dataset = SceneFlowData('train')
  elif FLAGS.mode == 'eval':
    dataset = SceneFlowData('validation')
  assert dataset.data_files()
  
  if tf.gfile.Exists(FLAGS.root_dir):
    tf.gfile.DeleteRecursively(FLAGS.root_dir)
  tf.gfile.MakeDirs(FLAGS.root_dir)

  FLAGS.train_dir = os.path.join(FLAGS.log_root, 'train')
  FLAGS.eval_dir = os.path.join(FLAGS.log_root, 'eval')

  hps = gcnet_model.HParams(batch_size=batch_size,
#                             min_lrn_rate=0.0001,
                             lrn_rate=0.01,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='RMSProp',
                             max_disparity=200) 

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps, dataset)
    elif FLAGS.mode == 'eval':
#      evaluate(hps, dataset)
      pass


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
