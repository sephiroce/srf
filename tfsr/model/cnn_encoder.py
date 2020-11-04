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
# citation: Ying Zhang, Mohammad Pezeshki, Philemon Brakel, Saizheng Zhang,
Cesar Laurent Yoshua Bengio, Aaron Courville, Towards End-to-End Speech
Recognition with Deep Convolutional Neural Networks, interspeech 2016
# arxiv: https://arxiv.org/abs/1701.02720
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import tensorflow as tf
import tfsr.helper.model_helper as mh
from tensorflow_addons.layers import Maxout

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

    #self.mask = tf.keras.layers.Lambda(mh.feat_mask, name="pad_mask1")
    self.mask2 = tf.keras.layers.Lambda(mh.feat_mask2, name="pad_mask2")

    self.model = tf.keras.models.Sequential()
    for i in range(4):
      self.model.add(tf.keras.layers.Conv2D(filters=self.nfilt_inp,
                                            kernel_size=(5, 3),
                                            activation='linear',
                                            padding='same', strides=1,
                                            kernel_initializer=mh.get_init(init),
                                            name="inn_conv1_%d"%(i+1),
                                            use_bias=False))
      self.model.add(tf.keras.layers.Dropout(rate=0.2,
                                             name="mo_drop_%d"%(i+1)))
      self.model.add(Maxout(num_units=self.nfilt_inp // 2,
                            name="maxout%d"%(i + 1)))
      if i == 0:
        self.model.add(tf.keras.layers.MaxPooling2D((1, 3)))
      self.model.add(tf.keras.layers.Dropout(rate=0.3,
                                             name="cnn_drop_%d"%(i+1)))
      self.model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6,
                                                        name="inn_ln_%d"%(i+1)))

    for i in range(4, enc_num - 1):
      self.model.add(tf.keras.layers.Conv2D(filters=self.nfilt_inn,
                                            kernel_size=(5, 3),
                                            activation='linear',
                                            padding='same',
                                            strides=1,
                                            kernel_initializer=mh.get_init(init),
                                            name="inn_conv1_%d"%(i+1),
                                            use_bias=False))
      self.model.add(tf.keras.layers.Dropout(rate=0.2,
                                             name="inn_drop1_%d"%(i+1)))
      self.model.add(Maxout(num_units=self.nfilt_inn // 2,
                            name="maxout%d" % (i + 1)))
      self.model.add(tf.keras.layers.Dropout(rate=0.3,
                                             name="cnn_drop_%d"%(i + 1)))
      self.model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6,
                                                        name="inn_ln_%d"%(i + 1)))

    feat_dim = config.feat_dim // 3
    last_filt = (self.proj_dim // feat_dim) * 2
    self.model.add(tf.keras.layers.Conv2D(filters=last_filt,
                                          kernel_size=(5, 3),
                                          activation='linear',
                                          padding='same',
                                          strides=1,
                                          kernel_initializer=mh.get_init(init),
                                          name="inn_conv1_%d"%(enc_num),
                                          use_bias=False))
    self.model.add(Maxout(num_units=last_filt // 2,
                          name="maxout%d" % (enc_num)))
    self.model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6,
                                                      name="inn_ln_%d"%(enc_num)))

    #Maxout Feedforward layers
    self.model.add(tf.keras.layers.Reshape((-1, feat_dim * (last_filt // 2))))

    for i in range(self.enc_num, self.enc_num + self.proj_layers - 1):
      self.model.add(tf.keras.layers.TimeDistributed(
          tf.keras.layers.Dense(self.proj_dim,
                                kernel_initializer=mh.get_init(init),
                                name="proj1_%d"%(i+1), use_bias=False)))
      self.model.add(tf.keras.layers.Dropout(rate=0.2, name="proj_drop_%d"%(i+1)))
      self.model.add(Maxout(self.proj_dim // 2, name="proj_maxout%d"%(i+1)))
      self.model.add(tf.keras.layers.Dropout(rate=0.3,
                                             name="proj_drop1_%d"%(i+1)))
      self.model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6,
                                                        name="proj_ln_%d"%(i+1)))

    #Maxout Last Projection layers
    self.model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(class_n * 2, kernel_initializer=mh.get_init(init),
                              use_bias=False)))
    self.model.add(tf.keras.layers.Dropout(rate=0.2,
                                           name="proj_drop_%d"%(self.proj_layers - 1)))
    self.model.add(Maxout(class_n, name="proj_maxout%d"%(self.proj_layers - 1)))
    self.model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6))

    logger.info("CNN CTC model, please check config model_conv_*")

  def call(self, inputs, **kwargs):
    # pylint: disable=arguments-differ
    inp_len, _ = kwargs["input_lengths"], kwargs["training"]

    return self.mask_layer(self.mask2([self.model(tf.expand_dims(inputs, -1)),
                                       inp_len, 1]))
