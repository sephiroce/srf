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
# pylint: disable=too-many-arguments, too-many-instance-attributes, invalid-name

"""
lstm_encoder.py: the implementation of lstm ctc
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import math
import tensorflow as tf
import tfsr.helper.model_helper as mh
from tfsr.model.sequence_router import CapsulationLayer

class LstmEncoder(tf.keras.Model): #pylint: disable=too-many-ancestors
  """
  An implementation of LSTM based speech encoders.
  """
  def get_config(self):
    pass

  def __init__(self, config, vocab_n):
    super().__init__()
    self.mask = tf.keras.layers.Lambda(mh.feat_mask2, name="pad_mask")

    num_layers = config.model_encoder_num
    d_model = config.model_dimension
    input_dropout = config.train_inp_dropout
    inner_dropout = config.train_inn_dropout
    init = config.model_initializer

    self.d_model = d_model
    self.num_layers = num_layers

    if config.model_type.lower() == "blstm":
      self.enc_layers = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
          d_model, return_sequences=True, kernel_initializer=mh.get_init(
              init)), merge_mode="ave") for _ in range(num_layers)]
    else:
      self.enc_layers = \
        [tf.keras.layers.LSTM(d_model, return_sequences=True,
                              kernel_initializer=mh.get_init(init))
         for _ in range(num_layers)]
    self.layernorms = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                       for _ in range(num_layers)]
    self.dropouts = [tf.keras.layers.Dropout(rate=inner_dropout)
                     for _ in range(num_layers)]
    self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.mask_layer = tf.keras.layers.Masking(mask_value=0.0)
    self.input_dropout = tf.keras.layers.Dropout(rate=input_dropout)
    self.proj = tf.keras.layers.Dense(
        vocab_n, kernel_initializer=mh.get_init(init), use_bias=False)

    kernel_size = 3
    self.stride = stride = config.model_conv_stride
    self.cnn_n = cnn_n = config.model_conv_layer_num
    self.feat_dim = math.ceil(config.feat_dim / (stride ** cnn_n))
    self.nfilt = nfilt = config.model_conv_filter_num

    self.conv = CapsulationLayer(cnn_n, nfilt, kernel_size, self.stride, init,
                                 name="conv_feat") \
                if config.model_lstm_is_cnnfe else None
    self.in_len_div = stride ** cnn_n if config.model_lstm_is_cnnfe else 1

  def call(self, embeddings, **kwargs):
    # pylint: disable=arguments-differ
    inp_len = kwargs["input_lengths"]
    training = kwargs["training"]

    if self.conv is not None:
      embeddings, batch, seq_len = self.conv(embeddings, input_lengths=inp_len)
      embeddings = tf.reshape(embeddings,
                              [batch, seq_len, self.feat_dim * self.nfilt],
                              name="reshape_conv")

    embeddings = self.input_dropout(embeddings, training=training)

    for idx, enc_layer in enumerate(self.enc_layers):
      embeddings = enc_layer(embeddings)
      embeddings = self.layernorms[idx](embeddings)
      embeddings = self.dropouts[idx](embeddings, training=training)

    embeddings = self.proj(embeddings)
    embeddings = self.mask([embeddings, inp_len, self.in_len_div])
    embeddings = self.mask_layer(embeddings)
    return self.ln(embeddings)
