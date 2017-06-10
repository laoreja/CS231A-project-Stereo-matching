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
"""Small library that points to the KITTI (combining 2012 and 2015) data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import Dataset

FLAGS = tf.app.flags.FLAGS

#Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '/home/laoreja/dataset/tfrecords_KITTI_merged',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")


class KITTIData(Dataset):
    """KITTI data set."""

    def __init__(self, subset):
        super(KITTIData, self).__init__('KITTI', subset, FLAGS.data_dir)

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        # Bounding box data consists of 615299 bounding boxes for 544546 images.
        if self.subset == 'train':
            return 394
        if self.subset == 'test':
            return 395



