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

""" Point wise feed forward network """

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import tensorflow as tf
import tfsr.helper.model_helper as mh

class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, ff_dropout, init):
    super(PointWiseFeedForwardNetwork, self).__init__()
    self.ff_relu = tf.keras.layers.Dense(dff, activation='relu', use_bias=True,
                                         kernel_initializer=mh.get_init(init))
    self.ff_dropout = tf.keras.layers.Dropout(rate=ff_dropout)
    self.ff_proj = tf.keras.layers.Dense(d_model, activation="linear",
                                         use_bias=True,
                                         kernel_initializer=mh.get_init(init))

  def call(self, inputs, **kwargs):
    is_training = kwargs["is_training"]
    ff_output = self.ff_relu(inputs)
    ff_output = self.ff_dropout(ff_output, training=is_training)
    return self.ff_proj(ff_output)

def main():
  # Testing feed forward network
  print("Test position-wise feed forward")
  sample_ffn = PointWiseFeedForwardNetwork(256, 2048, 0.1, "glorot")
  print("Output shape of Point wise feed forward network")
  print(sample_ffn(tf.random.uniform([64, 50, 33]), is_training=True).shape)

if __name__ == "__main__":
  main()
