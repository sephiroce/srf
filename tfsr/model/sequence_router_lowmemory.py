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

import math
import tensorflow as tf
import tfsr.helper.model_helper as mh
from tfsr.model.sequence_router import CapsulationLayer, squash, length


class SequenceRouter(tf.keras.Model):
  # pylint: disable=too-many-ancestors, invalid-name, arguments-differ
  # pylint: disable=unused-argument
  """
  SequentialRouter: Currently, it supports dynamic routing based sequence router
  """

  def __init__(self, config, logger, class_n):
    # pylint: disable=too-many-statements
    super(SequenceRouter, self).__init__()

    init = config.model_initializer
    kernel_size = 3
    self.stride = stride = 2
    self.cnn_n = cnn_n = config.model_conv_layer_num
    self.feat_dim = math.ceil(config.feat_dim / (stride * cnn_n))
    self.nfilt = nfilt = config.model_conv_filter_num
    self.class_n = class_n

    self.enc_num = enc_num = config.model_encoder_num

    self.lpad = config.model_caps_window_lpad
    self.rpad = config.model_caps_window_rpad
    self.window = window = self.lpad + self.rpad + 1

    self.is_context = is_context = config.model_caps_context
    self.caps_inp_n = caps_inp_n = config.model_caps_primary_num
    caps_inp_in_n = caps_inp_n * window
    self.caps_inp_d = caps_inp_d = config.model_caps_primary_dim
    self.caps_cov_n = caps_cov_n = config.model_caps_convolution_num
    caps_cov_in_n = caps_cov_n * window
    self.caps_cov_d = caps_cov_d = config.model_caps_convolution_dim
    self.caps_cls_n = caps_cls_n = class_n
    self.caps_cls_d = caps_cls_d = config.model_caps_class_dim

    # Capsulation w/ bottleneck projection layers.
    self.conv = CapsulationLayer(cnn_n, nfilt, kernel_size, stride, init,
                                 name="conv_feat")
    self.proj_pe = tf.keras.layers.Dense(caps_inp_n, activation='linear',
                                         kernel_initializer=mh.get_init(init),
                                         name="flatten")
    self.mask = tf.keras.layers.Lambda(mh.feat_mask, name="pad_mask")
    self.ecs = [tf.keras.layers.Conv2D(filters=config.model_caps_primary_dim,
                                       kernel_size=3, activation='linear',
                                       padding='same', strides=1,
                                       kernel_initializer=mh.get_init(init),
                                       name="encaps%d"%(i+1)) for i in range(2)]
    self.ecd = [tf.keras.layers.Dropout(rate=0.2, name="do_encaps%d" % (i + 1))
                for i in range(2)]
    self.inp_dropout = tf.keras.layers.Dropout(rate=config.train_inp_dropout,
                                               name="do_input")

    # Dynamic Routing variables
    shape = None
    if enc_num > 1:# in_n, out_n, out_dim, in_dim
      shape = [(1, 1, caps_inp_in_n, caps_cov_n, caps_cov_d, caps_inp_d)]
      for _ in range(1, enc_num - 1):
        shape.append((1, 1, caps_cov_in_n, caps_cov_n, caps_cov_d, caps_cov_d))
      shape.append((1, 1, caps_cov_in_n, caps_cls_n, caps_cls_d, caps_cov_d))
    elif enc_num == 1:
      shape = [(1, 1, caps_inp_in_n, caps_cls_n, caps_cls_d, caps_inp_d)]

    self.wgt = [tf.Variable(tf.random.normal(shape=(1, s[2], s[3], s[4], s[5]),
                                             stddev=0.1, dtype=tf.float32),
                            trainable=True, name="W%d"%i)
                for i, s in enumerate(shape)]
    self.bias = [tf.Variable(tf.random.normal(shape=(1, s[2], s[3], s[4], 1),
                                              stddev=0.1, dtype=tf.float32),
                             trainable=True, name="b%d"%i)
                 for i, s in enumerate(shape)]
    self.ln_i = tf.keras.layers.LayerNormalization(name="ln_input")
    self.ln_m = [tf.keras.layers.LayerNormalization(name="ln_mid%d" % (i + 1))
                 for i in range(enc_num)]
    self.ln_o = tf.keras.layers.LayerNormalization(name="ln_output")

    logger.info("Layer x %d, Iter x 1 (fixed), Init %s, Win %d (l:%d, r:%d), "%
                (enc_num, "SDR" if is_context else "DR", self.window, self.lpad,
                 self.rpad))
    self.mid_dropout = \
      [tf.keras.layers.Dropout(rate=config.train_inn_dropout,
                               name="dropout_mid_%d"%i) for i in range(enc_num)]
    self.mask_layer = tf.keras.layers.Masking(mask_value=0)

  def get_config(self):
    return {
        "primary_capsule_number": tf.shape(self.wgt[0])[2],
        "primary_capsule_dimension": tf.shape(self.wgt[0])[5],
        "class_capsule_dimension": tf.shape(self.wgt[-1])[5]
    }

  def call(self, inputs, **kwargs):
    inp_len = kwargs["input_lengths"]
    training = kwargs["training"]
    lpad = self.lpad
    rpad = self.rpad
    window = self.window
    caps_in_d = tf.shape(self.wgt[0])[4]

    # Capsulation: feature sequences to primary capsule
    conv_out, batch, seq_len = self.conv(inputs, input_lengths=inp_len)

    emb = tf.reshape(conv_out, [batch, seq_len, self.feat_dim * self.nfilt],
                     name="reshape_emb1")
    emb = tf.expand_dims(self.proj_pe(emb), -1)
    emb = tf.math.maximum(self.ecd[0](self.ecs[0](emb)),
                          self.ecd[1](self.ecs[1](emb)))
    emb = self.mask([emb, inp_len, self.stride ** 2])
    emb = self.mask_layer(emb)
    emb = tf.reshape(emb, [batch, seq_len, self.caps_inp_n, caps_in_d],
                     name="reshape_emb2")
    emb = squash(emb, -1)
    emb = tf.reshape(emb, [batch, seq_len, self.caps_inp_n * caps_in_d],
                     name="reshape_lni1")
    emb = self.ln_i(emb)
    emb = tf.reshape(emb, [batch, seq_len, self.caps_inp_n, caps_in_d],
                     name="reshape_lni2")
    emb = self.inp_dropout(emb, training=training)

    # Contextual Dynamic Routing
    for i in range(self.enc_num):
      caps_in_n, caps_in_d = tf.shape(self.wgt[i])[1], tf.shape(self.wgt[i])[4]
      caps_out_n, caps_out_d = tf.shape(self.wgt[i])[2], tf.shape(self.wgt[i])[3]

      # windowing
      emb_pad = tf.keras.layers.ZeroPadding2D(padding=((lpad, rpad), (0, 0)))(emb)
      emb = tf.concat([emb_pad[:, i:i + tf.shape(emb)[1], :, :]
                       for i in range(window)], 2)

      # computing prediction vectors
      caps1_ex = tf.expand_dims(tf.expand_dims(emb, -1), 3)
      u_hat = tf.tile(caps1_ex, [1, 1, 1, caps_out_n, 1, 1])
      wgt = tf.tile(self.wgt[i], [batch, 1, 1, 1, 1])
      bias = tf.tile(self.bias[i], [batch, 1, 1, 1, 1])
      # routing algorithm
      if self.is_context:
        vs = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1,
                            dynamic_size=True)
        v_zero = tf.zeros([batch, 1, caps_out_n, caps_out_d, 1])
        _srf_loop_body = SequenceRouter.pad_body_context \
          if i == self.enc_num - 1 else SequenceRouter.body_context
        _, _, vs, _, _, _ = tf.while_loop(SequenceRouter._srf_cond,
                                          _srf_loop_body,
                                          [u_hat, 0, vs, v_zero, wgt, bias])
        emb = tf.reshape(vs.concat(), [seq_len, batch, caps_out_n, caps_out_d],
                         name="reshape_dr")
        emb = tf.transpose(emb, [1, 0, 2, 3])
      else: # Dynamic Routing
        b = tf.zeros([batch, seq_len, caps_in_n, caps_out_n, 1, 1],
                     dtype=tf.float32)
        if i == self.enc_num - 1:
          masking = tf.concat(
              [tf.ones([batch, seq_len, caps_in_n, 1, 1, 1], dtype=tf.float32) *
               -1e9, tf.zeros([batch, seq_len, caps_in_n, caps_out_n - 1, 1, 1],
                              dtype=tf.float32)], 3)
        else:
          masking = tf.zeros([batch, seq_len, caps_in_n, caps_out_n, 1, 1],
                             dtype=tf.float32)
        dummy = tf.zeros([batch, seq_len, 1, caps_out_n, caps_out_d, 1])
        args = [u_hat, b, tf.constant(0), dummy, 1, caps_in_n, masking]
        _, _, _, emb, _, _, _ = tf.while_loop(SequenceRouter._dr_cond,
                                              SequenceRouter._dr_loop_body, args)
        emb = tf.squeeze(emb, [2, 5])

      # layer normalization and dropout
      emb = tf.reshape(emb, [batch, seq_len, caps_out_n * caps_out_d],
                       name="reshape_lna%d"%(i+1))
      emb = self.ln_m[i](emb)
      emb = tf.reshape(emb, [batch, seq_len, caps_out_n, caps_out_d],
                       name="reshape_lnb%d"%(i+1))
      emb = self.mid_dropout[i](emb, training=training)

    return self.ln_o(length(emb, axis=-1))

  @staticmethod
  def _dr_cond(u_hat, b, counter, v, routing_iter, max_i, masking):
    # pylint: disable=unused-argument
    return tf.less(counter, routing_iter)

  @staticmethod
  def _dr_loop_body(u_hat, b, counter, v, routing_iter, max_i, masking):
    # pylint: disable=unused-argument
    b += masking
    c = tf.nn.softmax(b, axis=3)  # caps2_n, since routing to caps2_n
    s = tf.reduce_sum(tf.multiply(c, u_hat), axis=2, keepdims=True)
    v = squash(s, axis=-2)
    b += tf.matmul(u_hat, tf.tile(v, [1, 1, max_i, 1, 1, 1]), transpose_a=True)
    return u_hat, b, tf.add(counter, 1), v, routing_iter, max_i, masking

  @staticmethod
  def _srf_cond(u_hats, idx, vs, v, wgt, bias):
    # pylint: disable=unused-argument
    return tf.less(idx, tf.shape(u_hats)[1])

  @staticmethod
  def pad_body_context(u_hats, idx, vs, v, wgt, bias):
    u_hat = u_hats[:, idx, :, :, :, :]
    u_hat = tf.matmul(wgt, u_hat) + bias
    batch = tf.shape(u_hat)[0]
    max_i = tf.shape(u_hat)[1]
    caps2_n = tf.shape(u_hat)[2]
    masking = tf.concat(
        [tf.ones([batch, max_i, 1, 1, 1], dtype=tf.float32) * -1e9,
         tf.zeros([batch, max_i, caps2_n - 1, 1, 1], dtype=tf.float32)], 2)

    b = tf.matmul(u_hat, tf.tile(v, [1, max_i, 1, 1, 1]), transpose_a=True)
    b += masking
    c = tf.nn.softmax(b, axis=2)
    s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
    v = squash(s, axis=-2)
    vs = vs.write(idx, v)
    return u_hats, tf.add(idx, 1), vs, v, wgt, bias

  @staticmethod
  def body_context(u_hats, idx, vs, v, wgt, bias):
    u_hat = u_hats[:, idx, :, :, :, :]
    u_hat = tf.matmul(wgt, u_hat) + bias
    max_i = tf.shape(u_hat)[1]
    b = tf.matmul(u_hat, tf.tile(v, [1, max_i, 1, 1, 1]), transpose_a=True)
    c = tf.nn.softmax(b, axis=2)
    s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
    v = squash(s, axis=-2)
    vs = vs.write(idx, v)
    return u_hats, tf.add(idx, 1), vs, v, wgt, bias
