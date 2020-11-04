#-*- coding:utf-8 -*-
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

# pylint: disable=import-error, too-few-public-methods, too-many-locals
# pylint: disable=too-many-arguments, too-many-instance-attributes

"""
sequence_router.py: the implementation of sequence dynamic router
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import tensorflow as tf
import tfsr.helper.model_helper as mh

@tf.function
def squash(s, axis=-1, epsilon=1e-7):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
  safe_norm = tf.sqrt(squared_norm + epsilon)
  squash_factor = squared_norm / (1. + squared_norm)
  unit_vector = s / safe_norm
  return squash_factor * unit_vector


@tf.function
def length(s, axis=-1, epsilon=1e-7, keepdims=False):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
  return tf.sqrt(squared_norm + epsilon)


class CapsulationLayer(tf.keras.layers.Layer):
  def __init__(self, cnn_n, nfilt, kernel_size, stride, init, **kwargs):
    super(CapsulationLayer, self).__init__(**kwargs)

    self.cnn_n = cnn_n
    self.stride = stride
    self.conv_layers, self.dropouts = [], []
    self.maskings = []
    for _ in range(self.cnn_n):
      self.maskings.append(tf.keras.layers.Masking(mask_value=0))
      self.conv_layers.append([tf.keras.layers.Conv2D(filters=nfilt,
                                                      kernel_size=kernel_size,
                                                      activation='linear',
                                                      padding='same',
                                                      strides=stride,
                                                      kernel_initializer=\
                                                      mh.get_init(init))
                               for _ in range(2)])
      self.dropouts.append([tf.keras.layers.Dropout(rate=0.2)
                            for _ in range(2)])
    self.bn_layers = [tf.keras.layers.BatchNormalization(axis=-1)
                      for _ in range(cnn_n)]
    self.mask_layer = tf.keras.layers.Lambda(mh.feat_mask)


  def call(self, inputs, **kwargs):
    # pylint: disable=invalid-name
    inp_len = kwargs["input_lengths"]
    x = tf.expand_dims(inputs, axis=-1)
    # TODO: self.cnn_n /= 2 need to be an even number.
    for conv_idx in range(self.cnn_n):
      x = self.maskings[conv_idx](x)
      x1 = self.dropouts[0][conv_idx](self.conv_layers[0][conv_idx](x))
      x2 = self.dropouts[1][conv_idx](self.conv_layers[1][conv_idx](x))
      x = tf.math.maximum(x1, x2)
      x = self.mask_layer([x, inp_len, self.stride ** (conv_idx + 1)])
      x = self.bn_layers[conv_idx](x)
      x = self.mask_layer([x, inp_len, self.stride ** (conv_idx + 1)])
    return x, tf.shape(x)[0], tf.shape(x)[1]
