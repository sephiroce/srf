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

# pylint: disable=import-error, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes


"""
cnn_encoder.py: convolutional ctc network
"""

import math

import tensorflow as tf
import tfsr.helper.model_helper as mh

class CNNEncoder(tf.keras.Model):
  def get_config(self):
    pass

  def __init__(self, config, logger, class_n):
    super(CNNEncoder, self).__init__()

    init = config.model_initializer
    self.enc_num = enc_num = config.model_encoder_num
    self.nfilt_inp = config.model_conv_inp_nfilt
    self.nfilt_inn = config.model_conv_inn_nfilt
    self.proj_layers = config.model_conv_proj_num
    self.proj_dim = config.model_conv_proj_dim
    self.mask_layer = tf.keras.layers.Masking(mask_value=0.0)
    # filter : [time, frequency]

    self.mask = tf.keras.layers.Lambda(mh.feat_mask, name="pad_mask1")
    self.mask2 = tf.keras.layers.Lambda(mh.feat_mask2, name="pad_mask2")

    """
    Maxout Conv layers
    """
    self.enc_layers1 = [tf.keras.layers.Conv2D(filters=self.nfilt_inp,
                                               kernel_size=(5, 3),
                                               activation='linear',
                                               padding='same', strides=1,
                                               kernel_initializer=mh.get_init(init),
                                               name="inn_conv1_%d"%(i+1))
                        for i in range(3)]
    self.enc_layers2 = [tf.keras.layers.Conv2D(filters=self.nfilt_inp,
                                               kernel_size=(5, 3),
                                               activation='linear',
                                               padding='same', strides=1,
                                               kernel_initializer=mh.get_init(init),
                                               name="inn_conv2%d"%(i+1))
                        for i in range(3)]
    for i in range(3, enc_num):
      self.enc_layers1.append(tf.keras.layers.Conv2D(filters=self.nfilt_inn,
                                                     kernel_size=(5, 3),
                                                     activation='linear',
                                                     padding='same',
                                                     strides=1,
                                                     kernel_initializer=mh.get_init(init),
                                                     name="inn_conv1_%d"%(i+1)))
      self.enc_layers2.append(tf.keras.layers.Conv2D(filters=self.nfilt_inn,
                                                     kernel_size=(5, 3),
                                                     activation='linear',
                                                     padding='same',
                                                     strides=1,
                                                     kernel_initializer=mh.get_init(init),
                                                     name="inn_conv2%d"%(i+1)))
    self.dropouts1 = \
      [tf.keras.layers.Dropout(rate=0.2, name="inn_drop1_%d"%(i+1))
       for i in range(enc_num)]
    self.dropouts2 = \
      [tf.keras.layers.Dropout(rate=0.2, name="inn_drop2_%d"%(i+1))
       for i in range(enc_num)]
    self.layernorms = \
      [tf.keras.layers.LayerNormalization(epsilon=1e-6, name="inn_ln_%d"%(i+1))
       for i in range(enc_num)]
    self.cnn_dropout = [tf.keras.layers.Dropout(rate=0.1,
                                                name="proj_drop_%d"%(i+1))
                        for i in range(enc_num)]

    """
    Maxout Projection layers
    """
    self.reshape_to_maxout = \
      tf.keras.layers.Reshape((-1, math.ceil(config.feat_dim / 3) *
                               self.nfilt_inn), name="reshape_to_ffwd")
    self.proj1 = [tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(self.proj_dim,
                              kernel_initializer=mh.get_init(init),
                              name="proj1_%d"%(i+1), use_bias=False))
                  for i in range(self.proj_layers)]
    self.proj2 = [tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(self.proj_dim,
                              kernel_initializer=mh.get_init(init),
                              name="proj2_%d"%(i+1), use_bias=False))
                  for i in range(self.proj_layers)]
    self.dropproj1 = [tf.keras.layers.Dropout(rate=0.2, name="proj_drop1_%d"%(i+1))
                      for i in range(self.proj_layers)]
    self.dropproj2 = [tf.keras.layers.Dropout(rate=0.2, name="proj_drop2_%d"%(i+1))
                      for i in range(self.proj_layers)]
    self.proj_dropout = [tf.keras.layers.Dropout(rate=0.1,
                                                 name="proj_drop_%d"%(i+1))
                         for i in range(self.proj_layers)]
    self.layernorms_proj = \
      [tf.keras.layers.LayerNormalization(epsilon=1e-6, name="proj_ln_%d"%(i+1))
       for i in range(self.proj_layers)]

    """
    Maxout Last Projection layers
    """
    self.projv1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(class_n, kernel_initializer=mh.get_init(init),
                              use_bias=False))
    self.projv2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(class_n, kernel_initializer=mh.get_init(init),
                              use_bias=False))
    self.dropprojv1 = tf.keras.layers.Dropout(rate=0.2)
    self.dropprojv2 = tf.keras.layers.Dropout(rate=0.2)
    self.layernorms_projv = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.pool = tf.keras.layers.MaxPooling2D((1, 3))
    logger.info("CNN CTC model, please check config model_conv_*")

  def call(self, inputs, **kwargs):
    # pylint: disable=arguments-differ
    inp_len = kwargs["input_lengths"]
    training = kwargs["training"]

    emb = tf.expand_dims(inputs, -1)

    # conv layers
    for idx in range(self.enc_num):
      emb1 = self.dropouts1[idx](self.enc_layers1[idx](emb))
      emb2 = self.dropouts2[idx](self.enc_layers2[idx](emb))
      emb = tf.math.maximum(emb1, emb2)
      if idx == 0:
        emb = self.pool(emb)
      emb = self.layernorms[idx](emb)
      emb = self.cnn_dropout[idx](emb, training=training)
      emb = self.mask([emb, inp_len, 1])

    # fully connected layers
    emb = self.reshape_to_maxout(emb)
    for idx in range(self.proj_layers):
      emb1 = self.dropproj1[idx](self.proj1[idx](emb))
      emb2 = self.dropproj2[idx](self.proj2[idx](emb))
      emb = tf.math.maximum(emb1, emb2)
      emb = self.layernorms_proj[idx](emb)
      emb = self.proj_dropout[idx](emb, training=training)
      emb = self.mask2([emb, inp_len, 1])

    # a projection layer
    emb1 = self.dropprojv1(self.projv1(emb))
    emb2 = self.dropprojv2(self.projv2(emb))
    emb = self.layernorms_projv(tf.math.maximum(emb1, emb2))
    emb = self.mask2([emb, inp_len, 1])
    return self.mask_layer(emb)
