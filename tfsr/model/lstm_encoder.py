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
lstm_encoder.py: the implementation of lstm ctc
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import tensorflow as tf
import tfsr.helper.model_helper as mh

class LstmEncoder(tf.keras.Model):
  def __init__(self, num_layers, d_model, input_dropout, inner_dropout,
               bidirecitonal, init, vocab_n):
    super(LstmEncoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.input_dropout = tf.keras.layers.Dropout(rate=input_dropout)

    if bidirecitonal:
      self.enc_layers = [tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(d_model, return_sequences=True,
                               kernel_initializer=mh.get_init(init)),
          merge_mode="ave") for _ in range(num_layers)]
    else:
      self.enc_layers = \
        [tf.keras.layers.LSTM(d_model, return_sequences=True,
                              kernel_initializer=mh.get_init(init))
         for _ in range(num_layers)]

    self.layernorms = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                       for _ in range(num_layers)]
    self.dropouts = [tf.keras.layers.Dropout(rate=inner_dropout)
                     for _ in range(num_layers)]
    self.proj = tf.keras.layers.Dense(vocab_n, use_bias=False,
                                      kernel_initializer=mh.get_init(init))
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.mask = tf.keras.layers.Lambda(mh.feat_mask2, name="pad_mask")
    self.mask_layer = tf.keras.layers.Masking(mask_value=0.0)

  def get_config(self):
    pass

  def call(self, embeddings, **kwargs):
    # pylint: disable=arguments-differ
    inp_len = kwargs["input_lengths"]
    training = kwargs["training"]

    embeddings = self.input_dropout(embeddings, training=training)

    for idx, enc_layer in enumerate(self.enc_layers):
      embeddings = enc_layer(embeddings)
      embeddings = self.layernorms[idx](embeddings)
      embeddings = self.dropouts[idx](embeddings, training=training)

    embeddings = self.layernorm(self.proj(embeddings))
    embeddings = self.mask([embeddings, inp_len, 1])
    embeddings = self.mask_layer(embeddings)
    return embeddings
