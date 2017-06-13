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
GC-Net Eval module.
Using multiple GPUs.
"""
import time
import six
import sys, os
from datetime import datetime
import re

import numpy as np
import tensorflow as tf

import gcnet_model_no_mask
import gcnet_model
#import image_processing_KITTI
#from KITTI_data import KITTIData
import image_processing_no_mask
from SceneFlow_data import SceneFlowData

BATCH_SIZE = 1

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_string('dataset', 'SceneFlow', '')
tf.app.flags.DEFINE_string('log_root', '',
                                                     'Directory to keep the checkpoints. Should be a '
                                                     'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 3,
                                                        'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('max_steps', 1614, 'max steps for evaluation')                            
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#tf.app.flags.DEFINE_string('mode', 'eval', 'train, resume, retrain')
tf.app.flags.DEFINE_boolean('debug', False,
                            """Whether to show verbose summaries.""")
tf.app.flags.DEFINE_string('ckpt_path', '/home/laoreja/tf/log/gcnet_retrain_6/train', "path of ckpt for resume training")      
tf.app.flags.DEFINE_boolean('with_mask', False,
                            """Whether to use masks.""")
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # cannot see print!
# TODO: merge the mask/no mask, different dataset versions.


def tower_loss(scope, hps, dataset):
    """
    Calculate the total loss on a single tower running the GC-Net model.

    Args:
        scope: unique prefix string identifying the GC-Net tower, e.g. 'tower_0'
        hps: hyper parameters to pass into the model
    Returns:
         Tensor of shape [] containing the total loss for a batch of data
    """
    # Get inputs.
    num_preprocess_threads = FLAGS.num_preprocess_threads
    left_images, right_images, disparitys = image_processing_no_mask.inputs(
                                    dataset,
                                    batch_size = hps.batch_size,
                                    num_preprocess_threads=num_preprocess_threads)

    # Build inference Graph.
    model = this_model.GCNet(hps, left_images, right_images, disparitys    , 'eval') 
    model.build_graph_to_loss()
    
    return model.abs_loss, model.larger_than_3px, model.larger_than_5px, model.larger_than_7px, model.variables_to_restore #model.summaries
    
    
def evaluate(hps, dataset):
    """Evaluation."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
                        
        tower_losses = []
        tower_3px = []
        tower_5px = []
        tower_7px = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (this_model.TOWER_NAME, i)) as scope:
                        if i == 0:
                            loss, px3, px5, px7, total_variables_to_restore = tower_loss(scope, hps, dataset)
                        else:
                            loss, px3, px5, px7, _ = tower_loss(scope, hps, dataset)
                        tower_losses.append(loss)
                        tower_3px.append(px3)
                        tower_5px.append(px5)
                        tower_7px.append(px3)
                        
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        
                        # Retain the summaries from the final tower.
#            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        loss = tf.reduce_mean(tf.stack(tower_losses))
        px3 = tf.reduce_mean(tf.stack(tower_3px))
        px5 = tf.reduce_mean(tf.stack(tower_5px))
        px7 = tf.reduce_mean(tf.stack(tower_7px))
        
        
        # Add a summary to track the learning rate.
#    summaries.append(model_summary)
        
                    
        # Track the moving averages of all trainable variables.
        
        # Build the summary operation from the last tower summaries.
#    summary_op = tf.summary.merge(summaries)
        
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        
        tf.logging.info("before sess init")
        sess.run(init)

        tf.logging.info("before read ckpt")    
        restorer = tf.train.Saver(total_variables_to_restore)
        restorer.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_path))
        tf.logging.info('%s: Pre-trained model restored from %s' %
                    (datetime.now(), FLAGS.ckpt_path))


        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        
#    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)
        avg_loss = 0.0
        avg_px3 = 0.0
        avg_px5 = 0.0
        avg_px7 = 0.0
        for step in xrange(FLAGS.max_steps):     
            got_loss, got_px3, got_px5, got_px7 = sess.run([loss, px3, px5, px7])
            avg_loss += got_loss
            avg_px3 += got_px3
            avg_px5 += got_px5
            avg_px7 += got_px7
            
#      if step % 30 == 0:
#        summary_str = sess.run(summary_op)
#        summary_writer.add_summary(summary_str, step)
            tf.logging.info('step: %d, abs loss: %f, px3, 5, 7: %f %f %f' % (step, got_loss, got_px3, got_px5, got_px7))
                
        avg_loss /= (step*1.0)
        avg_px3 /= (step*1.0)
        avg_px5 /= (step*1.0)
        avg_px7 /= (step*1.0)
        tf.logging.info('final avg results: %f %f %f %f' % (avg_loss, avg_px3, avg_px5, avg_px7))

def main(_):
    if FLAGS.num_gpus == 0:
        raise ValueError('Only support multi gpu.')

    dataset = SceneFlowData('validation')
#  dataset = KITTIData('train')
    assert dataset.data_files()

    FLAGS.eval_dir = os.path.join(FLAGS.log_root, 'eval')
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    
    global this_model
    if FLAGS.with_mask:
        this_model = gcnet_model
    else:
        this_model = gcnet_model_no_mask

    hps = this_model.HParams(batch_size=BATCH_SIZE,
                                                         lrn_rate=0.0,
                                                         weight_decay_rate=0.0,
                                                         relu_leakiness=0.1,
                                                         optimizer='RMSProp',
                                                         max_disparity=192) 
    print("before train")
    evaluate(hps, dataset)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
