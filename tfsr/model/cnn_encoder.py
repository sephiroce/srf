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
# reference: Ying Zhang, Mohammad Pezeshki, Philemon Brakel, Saizheng Zhang,
Cesar Laurent Yoshua Bengio, Aaron Courville, Towards End-to-End Speech
Recognition with Deep Convolutional Neural Networks, interspeech 2016
# arxiv: https://arxiv.org/abs/1701.02720
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import tensorflow as tf
import tfsr.helper.model_helper as mh

class CNNEncoder(tf.keras.Model):
  def get_config(self):
    pass

  def __init__(self, config, logger, class_n):
    super(CNNEncoder, self).__init__()

    pool_freq = 3
    init = config.model_initializer
    self.enc_num = enc_num = config.model_encoder_num
    self.nfilt_inp = config.model_conv_inp_nfilt
    self.nfilt_inn = config.model_conv_inn_nfilt
    self.proj_layers = config.model_conv_proj_num
    self.proj_dim = config.model_conv_proj_dim
    self.mask_layer = tf.keras.layers.Masking(mask_value=0.0)
    self.stride = config.model_conv_stride
    # filter : [time, frequency]

    self.mask = tf.keras.layers.Lambda(mh.feat_mask, name="pad_mask1")
    self.mask2 = tf.keras.layers.Lambda(mh.feat_mask2, name="pad_mask2")
    assert config.model_conv_layer_num < 4

    # Maxout Conv layers
    self.enc_layers = [tf.keras.layers.Conv2D(filters=self.nfilt_inp,
                                              kernel_size=(5, 3),
                                              activation='linear',
                                              padding='same', strides=(self.stride, 1),
                                              kernel_initializer=mh.get_init(init),
                                              name="inn_conv1_%d"%(i + 1),
                                              use_bias=False)
                       for i in range(config.model_conv_layer_num)]

    for i in range(4 - config.model_conv_layer_num):
      self.enc_layers.append(
          tf.keras.layers.Conv2D(filters=self.nfilt_inp,
                                 kernel_size=(5, 3),
                                 activation='linear',
                                 padding='same', strides=1,
                                 kernel_initializer=mh.get_init(init),
                                 name="inn_conv1_%d"%(i + 3),
                                 use_bias=False))

    for i in range(4, enc_num - 1):
      self.enc_layers.append(tf.keras.layers.Conv2D(filters=self.nfilt_inn,
                                                    kernel_size=(5, 3),
                                                    activation='linear',
                                                    padding='same',
                                                    strides=1,
                                                    kernel_initializer=mh.get_init(init),
                                                    name="inn_conv1_%d"%(i+1),
                                                    use_bias=False))

    #TODO: Hardcoded 3
    feat_dim = config.feat_dim // pool_freq
    last_filt = (self.proj_dim // feat_dim) * 2
    self.enc_layers.append(tf.keras.layers.Conv2D(filters=last_filt,
                                                  kernel_size=(5, 3),
                                                  activation='linear',
                                                  padding='same',
                                                  strides=1,
                                                  kernel_initializer=mh.get_init(init),
                                                  name="inn_conv1_%d"%(enc_num - 1),
                                                  use_bias=False))

    self.dropouts = \
      [tf.keras.layers.Dropout(rate=0.2, name="inn_drop1_%d"%(i+1))
       for i in range(enc_num)]
    self.dropouts_cnn = \
      [tf.keras.layers.Dropout(rate=config.train_inn_dropout,
                               name="inn_drop1_%d"%(i+1))
       for i in range(enc_num)]
    self.layernorms = \
      [tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                          name="inn_ln_%d"%(i+1))
       for i in range(enc_num)]

    """
    Maxout Projection layers
    """
    self.reshape_to_maxout = \
      tf.keras.layers.Reshape((-1, feat_dim * (last_filt // 2)),
                              name="reshape_to_ffwd")
    self.proj = [tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(self.proj_dim,
                              kernel_initializer=mh.get_init(init),
                              name="proj1_%d"%(i+1), use_bias=False))
                 for i in range(self.proj_layers - 1)]
    self.dropproj = [tf.keras.layers.Dropout(rate=0.2,
                                             name="proj_drop1_%d"%(i+1))
                     for i in range(self.proj_layers - 1)]
    self.dropouts_proj = [tf.keras.layers.Dropout(rate=config.train_inn_dropout,
                                                  name="proj_drop1_%d"%(i+1))
                          for i in range(self.proj_layers - 1)]
    self.layernorms_proj = \
      [tf.keras.layers.LayerNormalization(epsilon=1e-6, name="proj_ln_%d"%(i+1))
       for i in range(self.proj_layers - 1)]

    """
    Maxout Last Projection layers
    """
    self.projv = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(class_n * 2, kernel_initializer=mh.get_init(init),
                              use_bias=False))
    self.dropprojv = tf.keras.layers.Dropout(rate=config.train_inn_dropout)
    self.layernorms_projv = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.pool = tf.keras.layers.MaxPooling2D((1, 3))
    logger.info("CNN CTC model, please check config model_conv_*, "
                "last_filt:%d", last_filt)

  def call(self, inputs, **kwargs):
    # pylint: disable=arguments-differ
    inp_len = kwargs["input_lengths"]
    training = kwargs["training"]

    emb = tf.expand_dims(inputs, -1)

    # conv layers
    for idx in range(self.enc_num):
      emb = self.dropouts[idx](self.enc_layers[idx](emb), training=training)
      dim = tf.cast(tf.shape(emb)[3] / 2, tf.int32)
      emb = tf.math.maximum(emb[:, :, :, :dim], emb[:, :, :, dim:])
      if idx == 0:
        emb = self.pool(emb)
      emb = self.layernorms[idx](emb)
      emb = self.dropouts_cnn[idx](emb)
      if idx == 0:
        emb = self.mask([emb, inp_len, self.stride])
      else:
        emb = self.mask([emb, inp_len, self.stride * self.stride])


    # fully connected layers
    emb = self.reshape_to_maxout(emb)
    for idx in range(self.proj_layers - 1):
      emb = self.dropproj[idx](self.proj[idx](emb), training=training)
      dim = tf.cast(tf.shape(emb)[2] / 2, tf.int32)
      emb = self.layernorms_proj[idx](tf.math.maximum(emb[:, :, :dim],
                                                      emb[:, :, dim:]))
      emb = self.dropouts_proj[idx](emb)
      emb = self.mask2([emb, inp_len, self.stride * self.stride])

    # a projection layer
    emb = self.dropprojv(self.projv(emb), training=training)
    dim = tf.cast(tf.shape(emb)[2] / 2, tf.int32)
    emb = self.layernorms_projv(tf.math.maximum(emb[:, :, :dim],
                                                emb[:, :, dim:]))
    emb = self.mask2([emb, inp_len, self.stride * self.stride])

    return self.mask_layer(emb)
