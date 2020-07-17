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
transformer.py: transformer model class
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import tensorflow as tf
import tfsr.helper.model_helper as mh
import tfsr.model.block as block
from tfsr.utils.beam_search_decoder import BeamSearchDecoder as bsd


##########################
# Create the Transformer #
##########################
class Transformer(tf.keras.Model):
  # pylint: disable=too-many-ancestors
  def __init__(self, config, dec_in_dim, dec_out_dim, ap=None):
    super(Transformer, self).__init__()

    encoder_num_layers = config.model_encoder_num
    decoder_num_layer = config.model_decoder_num
    d_model = config.model_dimension
    num_heads = config.model_att_head_num
    dff = config.model_inner_dim
    is_shared_embed = config.model_shared_embed
    input_dropout = config.train_inp_dropout
    inner_dropout = config.train_inn_dropout
    residual_dropout = config.train_res_dropout
    attention_dropout = config.train_att_dropout
    filter_number = config.model_conv_filter_num
    cnn_n = config.model_conv_layer_num
    conv_mask_type = config.model_conv_mask_type
    feat_dim = config.feat_dim
    emb_sqrt = config.model_emb_sqrt
    init = config.model_initializer
    self.enc_res_num = config.model_res_enc
    self.dec_res_num = config.model_res_dec

    self.encoder = Encoder(encoder_num_layers, d_model, num_heads, dff,
                           feat_dim, input_dropout, inner_dropout,
                           residual_dropout, attention_dropout, filter_number,
                           cnn_n, conv_mask_type, emb_sqrt, init)

    self.decoder = Decoder(decoder_num_layer, d_model, num_heads, dff,
                           dec_in_dim, dec_out_dim, input_dropout, inner_dropout,
                           residual_dropout, attention_dropout,
                           is_shared_embed, emb_sqrt, init)

    self.predict = Predictor(self.decoder, config.decoding_beam_width,
                             config.decoding_lp_alpha, ap, config.prep_max_tar)

  def call(self, inputs, **kwargs):
    #pylint: disable=arguments-differ
    dec_att_pen = kwargs["dec_att_pen"]
    dec_padding_mask = kwargs["dec_padding_mask"]
    enc_att_pen = kwargs["enc_att_pen"]
    enc_dec_att_pen = kwargs["enc_dec_att_pen"]
    enc_output = kwargs["enc_output"]
    enc_padding_mask = kwargs["enc_padding_mask"]
    inp_len = kwargs["input_length"]
    is_training = kwargs["is_training"]
    la_mask = kwargs["look_ahead_mask"]
    y_pred = kwargs["y_pred"]

    # (batch_size, inp_seq_len, d_model)
    if is_training or enc_output is None:
      computed_enc_output = self.encoder(inputs, input_lengths=inp_len,
                                         is_training=is_training,
                                         mask=enc_padding_mask,
                                         attention_penalty_mask=enc_att_pen,
                                         res_num=self.enc_res_num)
    else:
      computed_enc_output = enc_output

    if y_pred is None:
      dec_output, attention_weights = self.predict(computed_enc_output,
                                                   padding_mask=dec_padding_mask,
                                                   is_training=is_training)
    else:
      # dec_output.shape == (batch_size, tar_seq_len, d_model)
      dec_output, attention_weights = self.decoder((y_pred, computed_enc_output),
                                                   is_training=is_training,
                                                   look_ahead_mask=la_mask,
                                                   padding_mask=dec_padding_mask,
                                                   dec_att_pen=dec_att_pen,
                                                   enc_dec_att_pen=enc_dec_att_pen,
                                                   res_num=self.dec_res_num)

    return dec_output, attention_weights, computed_enc_output


###########
# Encoder #
###########
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, feat_dim,
               input_dropout, inner_dropout, residual_dropout,
               attention_dropout, filter_number, cnn_n, conv_mask_type,
               emb_sqrt, init):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.dff = dff
    self.num_layers = num_layers
    self.attention_dropout = attention_dropout
    self.num_heads = num_heads
    self.residual_dropout = residual_dropout
    self.emb_sqrt = emb_sqrt

    assert 0 <= conv_mask_type <= 2
    self.conv_mask_type = conv_mask_type

    self.enc_layers = [block.EncoderBlock(self.d_model, self.num_heads, dff,
                                          inner_dropout, residual_dropout,
                                          attention_dropout, init)
                       for _ in range(num_layers)]

    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    if self.conv_mask_type == 0 or self.conv_mask_type == 1:
      self.mask_layer = tf.keras.layers.Lambda(mh.feat_mask)
      self.mask2_layer = tf.keras.layers.Lambda(mh.feat_mask2)

    # TODO: stride and filter number are hard coded.
    self.stride = stride = 2
    kernel_size = 3
    self.cnn_n = cnn_n

    self.conv_layers = \
      [tf.keras.layers.Conv2D(filters=filter_number,
                              kernel_size=(kernel_size, kernel_size),
                              activation='relu', padding='same',
                              strides=(stride, stride),
                              kernel_initializer=mh.get_init(init)) for _ in
       range(cnn_n)]

    self.bn_layers = \
      [tf.keras.layers.BatchNormalization(axis=-1) for _ in range(cnn_n)]

    self.reshape_to_ffwd = \
      tf.keras.layers.Reshape((-1, int(feat_dim / (stride * cnn_n))
                               * filter_number))

    self.linear_projection = tf.keras.layers.Dense(d_model,
                                                   activation='linear',
                                                   kernel_initializer=mh.get_init(init))

    self.input_dropout = tf.keras.layers.Dropout(rate=input_dropout)

  def get_config(self):
    return {
        "model_size": self.d_model,
        "inner_size": self.dff,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
        "residual_dropout": self.residual_dropout
    }

  def call(self, inputs, **kwargs):
    # pylint: disable=arguments-differ
    """
    from the Speech-Transformer paper,
      We firstly stack two 3×3 CNN layers with stride 2 for both time and
      frequency dimensions to prevent the GPU memory overflow and produce the
      approximate hidden representation length with the character length.
    """
    input_lengths = kwargs["input_lengths"]
    training = kwargs["is_training"]
    mask = kwargs["mask"]
    att_penalty = kwargs["attention_penalty_mask"]
    res_num = kwargs["res_num"]

    # Conv2D feature sequence
    embeddings = tf.expand_dims(inputs, axis=-1)
    # TODO: self.cnn_n /= 2 need to be an even number.
    for conv_idx in range(self.cnn_n):
      embeddings = self.conv_layers[conv_idx](embeddings)
      if self.conv_mask_type == 0:
        embeddings = self.mask_layer([embeddings, input_lengths,
                                      (conv_idx + 1) * self.stride])
      embeddings = self.bn_layers[conv_idx](embeddings)
      if self.conv_mask_type == 0 or self.conv_mask_type == 1:
        embeddings = self.mask_layer([embeddings, input_lengths,
                                      (conv_idx + 1) * self.stride])

    embeddings = self.reshape_to_ffwd(embeddings)
    embeddings = self.linear_projection(embeddings)
    if self.conv_mask_type == 0 or self.conv_mask_type == 1:
      embeddings = self.mask2_layer([embeddings, input_lengths,
                                     self.cnn_n * self.stride])

    # adding positional encodings to prepared feature sequences
    seq_len = tf.shape(embeddings)[1]
    if self.emb_sqrt:
      embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    embeddings += mh.get_pos_enc(seq_len, self.d_model)
    embeddings = self.input_dropout(embeddings, training=training)

    pre_emb = None if res_num == 1 else tf.identity(embeddings)
    for idx, enc_layer in enumerate(self.enc_layers):
      embeddings = enc_layer((embeddings, pre_emb), is_training=training,
                             mask=mask, attention_penalty_mask=att_penalty)
      if res_num is not None and res_num > 1:
        if idx > 0 and idx % res_num == 0:
          pre_emb = tf.identity(embeddings)

    return self.layernorm(embeddings) # (batch_size, input_seq_len, d_model)


###########
# Decoder #
###########
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, dec_in_dim,
               dec_out_dim, input_dropout, inner_dropout, residual_dropout,
               attention_dropout, is_shared_embed, emb_sqrt, init):
    super(Decoder, self).__init__()

    self._config = (num_layers, d_model, num_heads, dff, dec_in_dim,
                    dec_out_dim, input_dropout, inner_dropout, residual_dropout,
                    attention_dropout, is_shared_embed, emb_sqrt, init)

    self._d_model = d_model
    self.num_layers = num_layers
    self.emb_sqrt = emb_sqrt
    self._dec_in_dim = dec_in_dim
    self._dec_out_dim = dec_out_dim
    self.embedding = tf.keras.layers.Embedding(dec_in_dim, d_model,
                                               mask_zero=True)
    self.dec_layers = [block.DecoderBlock(d_model, num_heads, dff,
                                          inner_dropout, residual_dropout,
                                          attention_dropout, init)
                       for _ in range(num_layers)]
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    if not is_shared_embed:
      self.linear = tf.keras.layers.Dense(dec_out_dim,
                                          kernel_initializer=mh.get_init(init))
    else:
      self.linear = None

    self.input_dropout = tf.keras.layers.Dropout(rate=input_dropout)

  @property
  def d_model(self):
    return self._d_model

  @property
  def dec_in_dim(self):
    return self._dec_in_dim

  @property
  def dec_out_dim(self):
    return self._dec_out_dim

  @property
  def config(self):
    return self._config

  def call(self, inputs, **kwargs):
    # pylint: disable=arguments-differ
    input_vector, enc_out = inputs
    training = kwargs["is_training"]
    look_ahead_mask = kwargs["look_ahead_mask"]
    padding_mask = kwargs["padding_mask"]
    dec_att_pen = kwargs["dec_att_pen"]
    enc_dec_att_pen = kwargs["enc_dec_att_pen"]
    res_num = kwargs["res_num"]

    attention_weights = {}

    # (batch_size, target_seq_len, d_model)
    embeddings = self.embedding(input_vector)
    if self.emb_sqrt:
      embeddings *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
    seq_len = tf.shape(embeddings)[1]
    embeddings += mh.get_pos_enc(seq_len, self._d_model)
    embeddings = self.input_dropout(embeddings, training=training)

    pre_emb = None if res_num == 1 else tf.identity(embeddings)
    for idx, dec_layer in enumerate(self.dec_layers):
      embeddings, block1, block2 = dec_layer((embeddings, pre_emb, enc_out),
                                             is_training=training,
                                             look_ahead_mask=look_ahead_mask,
                                             padding_mask=padding_mask,
                                             dec_att_pen=dec_att_pen,
                                             enc_dec_att_pen=enc_dec_att_pen)
      if res_num is not None and res_num > 1:
        if idx > 0 and idx % res_num == 0:
          pre_emb = tf.identity(embeddings)

      attention_weights['decoder_layer{}_block1'.format(idx + 1)] = block1
      attention_weights['decoder_layer{}_block2'.format(idx + 1)] = block2

    # (batch_size, tar_seq_len, target_vocab_size)
    norm_decoder_output = self.layernorm(embeddings)

    # decoder_value.shape == (batch_size, target_seq_len, d_model)
    if self.linear is not None:
      output = self.linear(norm_decoder_output)
    else:
      output = tf.tensordot(norm_decoder_output,
                            tf.transpose(self.embedding.weights[0]), axes=1)

    return output, attention_weights


###########
# Predict #
###########
class Predictor(tf.keras.layers.Layer):
  def __init__(self, decoder, beam_width, lp_alpha, att_pen, max_dec_len,
               oh_inp=False):
    super(Predictor, self).__init__()
    self.decoder = decoder
    self.dec_in_dim = decoder.dec_in_dim
    self.bos_id = self.dec_out_dim = decoder.dec_out_dim
    self.eos_id = self.bos_id - 1
    self.beam_width = beam_width
    self.lp_alpha = lp_alpha
    self.att_pen = att_pen
    if max_dec_len > 0:
      self.max_dec_len = tf.cast(max_dec_len + max_dec_len * 0.1, tf.int32)
    else:
      self.max_dec_len = 250 # max length of wsj.
    self.d_model = decoder.d_caps
    self.oh_inp = oh_inp

  def _grow_paths(self, training, enc_out, dec_padding_mask, is_debug=False):
    """
    these parameters are not changed during decoding.
    :param training:
    :param enc_out: [batch, beam_width, tar_len, dim]
    :param dec_padding_mask: [batch, 1, 1, inp_len]
    :return: softmax output of decoding
    """
    def grow_path(tar, i, batch_size, beam_size, tar_len):
      position = i + 1

      # flatten
      if is_debug:
        tf.print("1. unflat y_pred:", tf.shape(tar))
      tar = tf.reshape(tar, [batch_size * beam_size, tar_len])
      look_ahead_mask = mh.create_combined_mask(tar)

      dec_att_pen = None if self.att_pen is None else self.att_pen[:, :position,
                                                                   :position]

      if is_debug:
        tf.print("2. flated y_pred:", tf.shape(tar))

      if self.oh_inp:
        tar = tf.one_hot(tar, self.dec_in_dim, on_value=1.0, off_value=0.0)

      #TODO: not supported stf_fs: tar = tf.one_hot(tar, len(str_to_int))
      #TODO: hard_coded enc_dec_att_pen is set to None, res_num is set to 1
      stf_output, _ = self.decoder((tar, enc_out), is_training=training,
                                   look_ahead_mask=look_ahead_mask,
                                   padding_mask=dec_padding_mask,
                                   dec_att_pen=dec_att_pen,
                                   enc_dec_att_pen=None,
                                   res_num=1)

      # unflatten
      unflat_out = tf.reshape(stf_output, [batch_size, beam_size, tar_len,
                                           self.dec_out_dim])
      if is_debug:
        tf.print("3. unflat decout:", tf.shape(unflat_out))
      return unflat_out

    return grow_path

  def call(self, inputs, **kwargs):
    """Return predicted sequences and their scores"""
    beam_width = self.beam_width
    vocab_size = self.dec_out_dim
    lp_alpha = self.lp_alpha
    max_dec_len = self.max_dec_len
    bos_id = self.bos_id
    eos_id = self.eos_id
    batch_size = tf.shape(inputs)[0]
    enc_dec_mask = kwargs["padding_mask"]
    is_training = kwargs["is_training"]

    inp_len = tf.shape(inputs)[1]
    enc_out = tf.expand_dims(inputs, axis=1)
    enc_out = tf.tile(enc_out, [1, self.beam_width, 1, 1])
    enc_out = tf.reshape(enc_out, [batch_size * beam_width, inp_len,
                                   self.d_model])

    tar_len = tf.shape(enc_dec_mask)[3]
    enc_dec_mask = tf.expand_dims(enc_dec_mask, axis=1)
    enc_dec_mask = tf.tile(enc_dec_mask, [1, self.beam_width, 1, 1, 1])
    enc_dec_mask = tf.reshape(enc_dec_mask, [batch_size * beam_width, 1, 1,
                                             tar_len])

    # return hypos ids and scores for computing E[WER], computing logits are
    # performed in the _growth_paths
    return bsd(self._grow_paths(is_training, enc_out, enc_dec_mask),
               vocab_size, batch_size, beam_width, lp_alpha, max_dec_len,
               bos_id, eos_id).search(is_training)


def main():
  # Test Encoder
  sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                           dff=2048, feat_dim=240, input_dropout=0.1,
                           inner_dropout=0.1, attention_dropout=0.1,
                           residual_dropout=0.1, filter_number=64, cnn_n=2,
                           conv_mask_type=0, init="glorot", emb_sqrt=True)

  temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

  sample_encoder_output = sample_encoder(temp_input, training=False, mask=None,
                                         input_lengths=None, att_penalty=None)
  print("Encoder output shape")
  print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

  # Test Decoder
  sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                           dff=2048, dec_in_dim=8000,
                           dec_out_dim=7999, input_dropout=0.1,
                           inner_dropout=0.1, attention_dropout=0.1,
                           residual_dropout=0.1, is_shared_embed=False,
                           init="glorot", emb_sqrt=True)

  temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

  output, attn = sample_decoder(temp_input,
                                enc_output=sample_encoder_output,
                                training=False,
                                look_ahead_mask=None,
                                padding_mask=None)

  print("Decoder output shape")
  print(output.shape)
  print("Decoder Layer2 Block2 output shape")
  print(attn['decoder_layer2_block2'].shape)

if __name__ == "__main__":
  main()
