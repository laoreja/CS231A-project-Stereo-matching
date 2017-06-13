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
"""Small library that points to the SceneFlow data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from dataset import Dataset

FLAGS = tf.app.flags.FLAGS

#Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '/home/laoreja/dataset/SceneFlow/tfrecords_SceneFlow',
                                    """Path to the processed data, i.e. """
                                    """TFRecord of Example protos.""")

class SceneFlowData(Dataset):
    """SceneFlow data set."""

    def __init__(self, subset):
#        self.data_dir = '/home/laoreja/dataset/SceneFlow/tfrecords_SceneFlow'
        super(SceneFlowData, self).__init__('SceneFlow', subset, FLAGS.data_dir)


    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train':
            return 34981
        if self.subset == 'validation':
            return 4843
        if self.subset == 'debug':
            return 7

