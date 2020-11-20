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
from tfsr.model.sequence_router import CapsulationLayer, squash

eps = 1e-9

class SequenceRouter(tf.keras.Model):
  # pylint: disable=too-many-ancestors, invalid-name, arguments-differ
  # pylint: disable=unused-argument
  """
  SequentialRouter: Currently, it supports dynamic routing based sequence router
  """

  def __init__(self, config, logger, class_n):
    # pylint: disable=too-many-statements
    super(SequenceRouter, self).__init__()

    init, kernel_size, self.stride = config.model_initializer, 3, 2
    self.feat_dim = math.ceil(config.feat_dim /
                              (self.stride * config.model_conv_layer_num))
    self.nfilt = config.model_conv_filter_num
    self.enc_num = config.model_encoder_num
    self.lpad = config.model_caps_window_lpad
    self.rpad = config.model_caps_window_rpad
    self.window = self.lpad + self.rpad + 1
    self.is_context = is_context = config.model_caps_context

    self.ph = config.model_caps_primary_num
    caps_inp_in_n = self.ph * self.window
    self.pd = config.model_caps_primary_dim
    self.caps_cov_n = caps_cov_n = config.model_caps_convolution_num
    caps_cov_in_n = caps_cov_n * self.window
    self.caps_cov_d = caps_cov_d = config.model_caps_convolution_dim
    self.caps_cls_n = caps_cls_n = class_n
    self.caps_cls_d = caps_cls_d = config.model_caps_class_dim
    self.iter = config.model_caps_iter

    # Capsulation w/ bottleneck projection layers.
    self.conv = CapsulationLayer(config.model_conv_layer_num, self.nfilt,
                                 kernel_size, self.stride, init)
    self.proj_pe = tf.keras.layers.Dense(self.ph, activation='linear',
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
    if self.enc_num > 1:# in_n, out_n, out_dim, in_dim
      shape = [(caps_inp_in_n, caps_cov_n, caps_cov_d, self.pd)]
      for _ in range(1, self.enc_num - 1):
        shape.append((caps_cov_in_n, caps_cov_n, caps_cov_d, caps_cov_d))
      shape.append((caps_cov_in_n, caps_cls_n, caps_cls_d, caps_cov_d))
    elif self.enc_num == 1:
      shape = [(caps_inp_in_n, caps_cls_n, caps_cls_d, self.pd)]

    self.wgt = [tf.Variable(tf.random.normal(shape=s, stddev=0.1,
                                             dtype=tf.float32), trainable=True,
                            name="W%d"%i) for i, s in enumerate(shape)]
    self.bias = [tf.Variable(tf.random.normal(shape=(1, 1, s[0], s[1], s[2]),
                                              stddev=0.1, dtype=tf.float32),
                             trainable=True, name="b%d"%i)
                 for i, s in enumerate(shape)]

    self.ln_i = tf.keras.layers.LayerNormalization(name="ln_input")
    self.ln_m = [tf.keras.layers.LayerNormalization(name="ln_mid%d" % (i + 1))
                 for i in range(self.enc_num)]
    self.ln_o = tf.keras.layers.LayerNormalization(name="ln_output")
    self.mid_dropout = [tf.keras.layers.Dropout(rate=config.train_inn_dropout,
                                                name="dropout_mid_%d"%i)
                        for i in range(self.enc_num)]
    self.mask_layer = tf.keras.layers.Masking(mask_value=0)

    logger.info("Layer x %d, Iter x %d, Init %s, Win %d (l:%d, r:%d)"%
                (self.enc_num, self.iter, "CONTEXT" if is_context else "ZERO",
                 self.window, self.lpad, self.rpad))
    logger.info("Transformation matrix size")
    size = 0
    for i, w in enumerate(self.wgt):
      logger.info("L=%d->%d"%(i, i+1))
      logger.info(tf.size(w))
      size += tf.size(w)
    logger.info("Total: %d"%size)

  def get_config(self):
    pass

  def call(self, inputs, **kwargs):
    inp_len = kwargs["input_lengths"]
    training = kwargs["training"]

    # Capsulation: feature sequences to primary capsule
    conv_out, batch, seq_len = self.conv(inputs, input_lengths=inp_len)
    emb = tf.reshape(conv_out, [batch, seq_len, self.feat_dim * self.nfilt])
    emb = self.proj_pe(emb)
    emb *= tf.math.sqrt(tf.cast(self.ph, tf.float32))
    emb += mh.get_pos_enc(seq_len, self.ph)
    emb = tf.expand_dims(emb, -1)
    emb = tf.math.maximum(self.ecd[0](self.ecs[0](emb)),
                          self.ecd[1](self.ecs[1](emb)))
    emb = self.mask([emb, inp_len, self.stride ** 2])
    emb = self.mask_layer(emb)
    emb = tf.reshape(emb, [batch, seq_len, self.ph, self.pd])
    emb = squash(emb, -1)
    emb = tf.reshape(emb, [batch, seq_len, self.ph * self.pd])
    emb = self.ln_i(emb)
    emb = tf.reshape(emb, [batch, seq_len, self.ph, self.pd])
    emb = self.inp_dropout(emb, training=training)

    # Contextual Dynamic Routing
    for i in range(self.enc_num):
      inh = tf.shape(self.wgt[i])[0]
      outh, outd = tf.shape(self.wgt[i])[1], tf.shape(self.wgt[i])[2]

      # windowing
      pemb = tf.keras.layers.ZeroPadding2D(padding=((self.lpad, self.rpad),
                                                    (0, 0)))(emb)
      emb = tf.concat([pemb[:, i:i + tf.shape(emb)[1], :, :]
                       for i in range(self.window)], 2)

      # routing algorithm
      u_hat = tf.einsum('ijkl,bsil->bsijk', self.wgt[i], emb) + \
              tf.tile(self.bias[i], [batch, seq_len, 1, 1, 1])
      if self.is_context:
        vs = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1,
                            dynamic_size=True)

        @tf.function
        def psdr(_u_hats, idx, _vs, _v):
          bat = tf.shape(_u_hats)[0]
          exp_i = tf.shape(_u_hats)[2]
          out_n = tf.shape(_u_hats)[3]
          pad_mask = tf.concat(
              [tf.ones([bat, exp_i, 1], dtype=tf.float32) * -1e9,
               tf.zeros([bat, exp_i, out_n - 1], dtype=tf.float32)], 2)

          _u_hat = _u_hats[:, idx, :, :, :]
          b = tf.einsum("bmij,bij->bmi", _u_hat, _v) + pad_mask
          c = tf.nn.softmax(b, axis=2)
          s = tf.reduce_sum(_u_hat * tf.expand_dims(c, -1), axis=1)
          _v = squash(s, axis=-1)
          for _ in range(1, self.iter):
            b += tf.einsum("bmij,bij->bmi", _u_hat, _v) + pad_mask
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(_u_hat * tf.expand_dims(c, -1), axis=1)
            _v = squash(s, axis=-1)
          _vs = _vs.write(idx, _v)
          return _u_hats, tf.add(idx, 1), _vs, _v

        @tf.function
        def sdr(_u_hats, idx, _vs, _v):
          _u_hat = _u_hats[:, idx, :, :, :]
          b = tf.einsum("bmij,bij->bmi", _u_hat, _v)
          c = tf.nn.softmax(b, axis=2)
          s = tf.reduce_sum(_u_hat * tf.expand_dims(c, -1), axis=1)
          _v = squash(s, axis=-1)
          for _ in range(1, self.iter):
            b += tf.einsum("bmij,bij->bmi", _u_hat, _v)
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(_u_hat * tf.expand_dims(c, -1), axis=1)
            _v = squash(s, axis=-1)
          _vs = _vs.write(idx, _v)
          return _u_hats, tf.add(idx, 1), _vs, _v

        _, _, vs, _ = \
          tf.while_loop(lambda a, b, c, d: tf.less(b, tf.shape(a)[1]),
                        psdr if i == self.enc_num - 1 else sdr,
                        [u_hat, 0, vs, tf.zeros([batch, outh, outd])])
        emb = tf.reshape(vs.concat(), [seq_len, batch, outh, outd])
        emb = tf.transpose(emb, [1, 0, 2, 3])
      else:
        b = tf.zeros([batch, seq_len, inh, outh, 1, 1], dtype=tf.float32)
        if i == self.enc_num - 1:
          masking = tf.concat(
              [tf.ones([batch, seq_len, inh, 1, 1, 1], dtype=tf.float32) * -1e9,
               tf.zeros([batch, seq_len, inh, outh - 1, 1, 1],
                        dtype=tf.float32)], 3)
        else:
          masking = tf.zeros([batch, seq_len, inh, outh, 1, 1], dtype=tf.float32)

        dummy = tf.zeros([batch, seq_len, 1, outh, outd, 1])
        args = [u_hat, b, tf.constant(0), dummy, self.iter, inh, masking]

        @tf.function
        def dr(u_hat, b, counter, v, routing_iter, max_i, masking):
          b += masking
          c = tf.nn.softmax(b, axis=3)  # caps2_n, since routing to caps2_n
          s = tf.reduce_sum(tf.multiply(c, u_hat), axis=2, keepdims=True)
          v = squash(s, axis=-2)
          b += tf.matmul(u_hat, tf.tile(v, [1, 1, max_i, 1, 1, 1]),
                         transpose_a=True)
          return u_hat, b, tf.add(counter, 1), v, routing_iter, max_i, masking

        _, _, _, emb, _, _, _ = \
          tf.while_loop(lambda a, b, c, d, r, f, g: tf.less(c, r), dr, args)
        emb = tf.squeeze(emb, [2, 5])

      # layer normalization and dropout
      emb = tf.reshape(emb, [batch, seq_len, outh * outd])
      emb = self.ln_m[i](emb)
      emb = tf.reshape(emb, [batch, seq_len, outh, outd])
      emb = self.mid_dropout[i](emb, training=training)

    return self.ln_o(tf.sqrt(tf.reduce_sum(tf.square(emb), axis=-1) + 1e-9))
