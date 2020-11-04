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
from tfsr.model.sequence_router import CapsulationLayer

eps = 1e-6

class SequenceRouter(tf.keras.Model):
  # pylint: disable=too-many-ancestors, invalid-name, arguments-differ
  # pylint: disable=unused-argument
  """
  SequentialRouter: Currently, it supports dynamic routing based sequence router
  """

  iter = -1

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

    self.iter = SequenceRouter.iter = config.model_caps_iter
    self.lpad = config.model_caps_window_lpad
    self.rpad = config.model_caps_window_rpad
    window = self.window = self.lpad + self.rpad + 1

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
    self.conv = CapsulationLayer(cnn_n, nfilt, kernel_size, stride, init, name="conv_feat")
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

    self.wgt = [tf.Variable(tf.random.normal(shape=s, stddev=0.1,
                                             dtype=tf.float32), trainable=True,
                            name="W%d"%i) for i, s in enumerate(shape)]
    self.bias = [tf.Variable(tf.random.normal(shape=(1, 1, s[2], s[3], s[4], 1),
                                              stddev=0.1, dtype=tf.float32),
                             trainable=True, name="b%d"%i)
                 for i, s in enumerate(shape)]
    self.ln_i = tf.keras.layers.LayerNormalization(name="ln_input")
    self.ln_m = [tf.keras.layers.LayerNormalization(name="ln_mid%d" % (i + 1))
                 for i in range(enc_num)]
    self.ln_o = tf.keras.layers.LayerNormalization(name="ln_output")

    logger.info("Layer x %d, Iter x %d, Init %s, Win %d (l:%d, r:%d), "
                % (enc_num, SequenceRouter.iter, "SDR" if is_context else
                   "DR", self.window, self.lpad, self.rpad))
    self.mid_dropout = \
      [tf.keras.layers.Dropout(rate=config.train_inn_dropout,
                               name="dropout_mid_%d"%i) for i in range(enc_num)]
    self.mask_layer = tf.keras.layers.Masking(mask_value=0)

  def get_config(self):
    pass

  def call(self, inputs, **kwargs):
    inp_len = kwargs["input_lengths"]
    training = kwargs["training"]
    lpad = self.lpad
    rpad = self.rpad
    window = self.window
    caps_in_d = tf.shape(self.wgt[0])[5]

    # Capsulation: feature sequences to primary capsule
    conv_out, batch, seq_len = self.conv(inputs, input_lengths=inp_len)

    emb = tf.reshape(conv_out, [batch, seq_len, self.feat_dim * self.nfilt], name="reshape_emb1")
    emb = tf.expand_dims(self.proj_pe(emb), -1)
    emb = tf.math.maximum(self.ecd[0](self.ecs[0](emb)), self.ecd[1](self.ecs[1](emb)))
    emb = self.mask([emb, inp_len, self.stride ** 2])
    emb = self.mask_layer(emb)
    emb = tf.reshape(emb, [batch, seq_len, self.caps_inp_n, caps_in_d], name="reshape_emb2")
    emb = SequenceRouter.squash(emb, -1)

    emb = tf.reshape(emb, [batch, seq_len, self.caps_inp_n * caps_in_d], name="reshape_lni1")
    emb = self.ln_i(emb)
    emb = tf.reshape(emb, [batch, seq_len, self.caps_inp_n, caps_in_d], name="reshape_lni2")
    emb = self.inp_dropout(emb, training=training)

    # Contextual Dynamic Routing
    for i in range(self.enc_num):
      caps_in_n, caps_in_d = tf.shape(self.wgt[i])[2], tf.shape(self.wgt[i])[5]
      caps_out_n, caps_out_d = tf.shape(self.wgt[i])[3], tf.shape(self.wgt[i])[4]

      # windowing
      emb_pad = tf.keras.layers.ZeroPadding2D(padding=((lpad, rpad), (0, 0)))(emb)
      emb = tf.concat([emb_pad[:, i:i + tf.shape(emb)[1], :, :] for i in range(window)], 2)

      # computing prediction vectors
      caps1_ex = tf.expand_dims(tf.expand_dims(emb, -1), 3)
      caps1_ex_tiled = tf.tile(caps1_ex, [1, 1, 1, caps_out_n, 1, 1])
      tiled_b = tf.tile(self.bias[i], [batch, seq_len, 1, 1, 1, 1])
      u_hat = tf.matmul(tf.tile(self.wgt[i], [batch, seq_len, 1, 1, 1, 1]), caps1_ex_tiled)
      u_hat = tf.reshape(u_hat, [batch, seq_len, caps_in_n, caps_out_n,
                                 caps_out_d, 1], name="reshape_emb3") + tiled_b

      # routing algorithm
      if self.is_context:
        vs = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True)
        v_zero = tf.zeros([batch, 1, caps_out_n, caps_out_d, 1])
        body_context = self.pad_body_context if i == self.enc_num - 1 else \
          self.body_context
        _, _, vs, _ = tf.while_loop(self.cond, body_context, [u_hat, 0, vs, v_zero])
        emb = tf.reshape(vs.concat(), [seq_len, batch, caps_out_n, caps_out_d],
                         name="reshape_dr")
        emb = tf.transpose(emb, [1, 0, 2, 3])
      else:
        b = tf.zeros([batch, seq_len, caps_in_n, caps_out_n, 1, 1], dtype=tf.float32)
        if i == self.enc_num - 1:
          masking = \
            tf.concat([tf.ones([batch, seq_len, caps_in_n, 1, 1, 1],
                               dtype=tf.float32) * -1e9,
                       tf.zeros([batch, seq_len, caps_in_n, caps_out_n - 1, 1, 1],
                                dtype=tf.float32)], 3)
        else:
          masking = tf.zeros([batch, seq_len, caps_in_n, caps_out_n, 1, 1],
                             dtype=tf.float32)
        dummy = tf.zeros([batch, seq_len, 1, caps_out_n, caps_out_d, 1])
        args = [u_hat, b, tf.constant(0), dummy, self.iter, caps_in_n, masking]
        _, _, _, emb, _, _, _ = tf.while_loop(self._condition, self._loop_body, args)
        emb = tf.squeeze(emb, [2, 5])

      # layer normalization and dropout
      emb = tf.reshape(emb, [batch, seq_len, caps_out_n * caps_out_d], name="reshape_lna%d"%(i+1))
      emb = self.ln_m[i](emb)
      emb = tf.reshape(emb, [batch, seq_len, caps_out_n, caps_out_d], name="reshape_lnb%d"%(i+1))
      emb = self.mid_dropout[i](emb, training=training)

    return self.ln_o(SequenceRouter.length(emb, axis=-1))

  @staticmethod
  def _condition(u_hat, b, counter, v, routing_iter, max_i, masking):
    return tf.less(counter, routing_iter)

  @staticmethod
  def _loop_body(u_hat, b, counter, v, routing_iter, max_i, masking):
    b += masking
    c = tf.nn.softmax(b, axis=3) # caps2_n, since routing to caps2_n
    s = tf.reduce_sum(tf.multiply(c, u_hat), axis=2, keepdims=True)
    v = SequenceRouter.squash(s, axis=-2)
    b += tf.matmul(u_hat, tf.tile(v, [1, 1, max_i, 1, 1, 1]), transpose_a=True)
    return u_hat, b, tf.add(counter, 1), v, routing_iter, max_i, masking

  @staticmethod
  def cond(u_hats, idx, vs, v):
    return tf.less(idx, tf.shape(u_hats)[1])

  @staticmethod
  def pad_body_context(u_hats, idx, vs, v):
    u_hat = u_hats[:, idx, :, :, :, :]
    batch = tf.shape(u_hat)[0]
    max_i = tf.shape(u_hat)[1]
    caps2_n = tf.shape(u_hat)[2]
    b = tf.zeros([batch, max_i, caps2_n, 1, 1], dtype=tf.float32)
    masking = tf.concat([tf.ones([batch, max_i, 1, 1, 1], dtype=tf.float32) * -1e9,
                         tf.zeros([batch, max_i, caps2_n-1, 1, 1], dtype=tf.float32)], 2)

    for _ in range(SequenceRouter.iter):
      b += tf.matmul(u_hat, tf.tile(v, [1, max_i, 1, 1, 1]), transpose_a=True)
      b += masking
      c = tf.nn.softmax(b, axis=2)
      s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
      v = SequenceRouter.squash(s, axis=-2)
    vs = vs.write(idx, v)
    return u_hats, tf.add(idx, 1), vs, v

  @staticmethod
  def body_context(u_hats, idx, vs, v):
    u_hat = u_hats[:, idx, :, :, :, :]
    batch = tf.shape(u_hat)[0]
    max_i = tf.shape(u_hat)[1]
    caps2_n = tf.shape(u_hat)[2]
    b = tf.zeros([batch, max_i, caps2_n, 1, 1], dtype=tf.float32)

    for _ in range(SequenceRouter.iter):
      b += tf.matmul(u_hat, tf.tile(v, [1, max_i, 1, 1, 1]), transpose_a=True)
      c = tf.nn.softmax(b, axis=2)
      s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
      v = SequenceRouter.squash(s, axis=-2)
    vs = vs.write(idx, v)
    return u_hats, tf.add(idx, 1), vs, v

  @staticmethod
  def squash(s, axis=-1, epsilon=1e-7):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector

  @staticmethod
  def length(s, axis=-1, epsilon=1e-7, keepdims=False):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
    return tf.sqrt(squared_norm + epsilon)
