from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import image_processing
from SceneFlow_data import SceneFlowData
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/home/laoreja/tf/log/test_input', '')

def train(dataset):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        num_preprocess_threads = FLAGS.num_preprocess_threads
#        left_images, right_images, disparitys, masks = image_processing.distorted_inputs(
#            dataset,
#            num_preprocess_threads=num_preprocess_threads)
        left_images, right_images, disparitys, masks = image_processing.inputs(
            dataset,
            num_preprocess_threads=num_preprocess_threads)

        summary_op = tf.summary.merge_all()
        
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            FLAGS.train_dir, graph=sess.graph)
            
        for step in range(10):
            start_time = time.time()
            a, b, c, d = sess.run([left_images, right_images, disparitys, masks])
            duration = time.time() - start_time

            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            duration = time.time() - start_time
            print('duration', duration)
#        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))


def main(_):
    dataset = SceneFlowData('train')
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train(dataset)


if __name__ == '__main__':
    tf.app.run()