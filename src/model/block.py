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
block.py: encoder or decoder block
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import tensorflow as tf
import tfsr.model.attention as att
import tfsr.helper.model_helper as mh
from tfsr.model.feed_forward import PointWiseFeedForwardNetwork


class EncoderBlock(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, inner_dropout, residual_dropout,
               attention_dropout, init):
    super(EncoderBlock, self).__init__()

    self.mha = att.MultiHeadAttention(d_model, num_heads, init)

    self.ffn = PointWiseFeedForwardNetwork(d_model, dff, inner_dropout, init)

    self.layernorm_cur = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm_pre = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm_res = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.res_dropout1 = tf.keras.layers.Dropout(residual_dropout)
    self.res_dropout2 = tf.keras.layers.Dropout(residual_dropout)

    self.attention_dropout = attention_dropout

  def call(self, inputs, **kwargs):
    is_training = kwargs["is_training"]
    mask = kwargs["mask"]
    attention_penalty_mask = kwargs["attention_penalty_mask"]

    emb = self.layernorm_cur(inputs)
    att_drop = self.attention_dropout if is_training else 0

    # (batch_size, input_seq_len, d_model)
    # input tuple: value, key, query
    attn_output, _ = self.mha((emb, emb, emb),
                              mask=mask, attention_dropout=att_drop,
                              attention_penalty_mask=attention_penalty_mask)
    attn_output = self.res_dropout1(attn_output, training=is_training)
    out1 = inputs + attn_output  # (batch_size, input_seq_len, d_model)

    nout1 = self.layernorm_res(out1)
    # (batch_size,  input_seq_len, d_model)
    ffn_output = self.ffn(nout1, is_training=is_training)
    ffn_output = self.res_dropout2(ffn_output, training=is_training)
    out2 = out1 + ffn_output  # (batch_size, input_seq_len, d_model)

    return out2


class EncoderMFBlock(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, inner_dropout, residual_dropout,
               attention_dropout, init):
    super(EncoderMFBlock, self).__init__()

    self.mha1 = att.MultiHeadAttention(d_model, num_heads, init)
    self.mha2 = att.MultiHeadAttention(d_model, num_heads, init)

    self.ffn = PointWiseFeedForwardNetwork(d_model, dff, inner_dropout, init)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm_raw = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.res_dropout1 = tf.keras.layers.Dropout(residual_dropout)
    self.res_dropout2 = tf.keras.layers.Dropout(residual_dropout)
    self.res_dropout3 = tf.keras.layers.Dropout(residual_dropout)

    self.attention_dropout = attention_dropout

    self.dense = tf.keras.layers.Dense(d_model, use_bias=False,
                                       kernel_initializer=mh.get_init(init))

  def call(self, inputs, **kwargs):
    raw_emb, feat_emb = inputs
    is_training = kwargs["is_training"]
    mask = kwargs["mask"]
    attention_penalty_mask = kwargs["attention_penalty_mask"]

    # Block1: Decoder input
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    normalized_input_vector = self.layernorm1(feat_emb)
    if raw_emb is None:
      normed_raw_emb = feat_emb
    else:
      normed_raw_emb = self.layernorm_raw(raw_emb)
    att_drop = self.attention_dropout if is_training else 0

    # (batch_size, target_seq_len, d_model)
    attn1, _ = self.mha1((normalized_input_vector, normalized_input_vector,
                          normalized_input_vector), mask=mask,
                         attention_dropout=att_drop,
                         attention_penalty_mask=attention_penalty_mask)

    attn1 = self.res_dropout1(attn1, training=is_training)
    out1 = attn1 + feat_emb

    # Block2: Encoder to Decoder input
    nout1 = self.layernorm2(out1)
    # (batch_size, target_seq_len, d_model)
    # input tuple: value, key, query
    attn2, _ = self.mha2((normed_raw_emb, normed_raw_emb, nout1), mask=mask,
                         attention_dropout=att_drop,
                         attention_penalty_mask=attention_penalty_mask)
    attn2 = self.res_dropout2(attn2, training=is_training)
    out2 = self.dense(attn2) + out1 # (batch_size, target_seq_len,
    # d_model)

    # Block3: Decoder output
    nout2 = self.layernorm3(out2)
    # (batch_size, target_seq_len, d_model)
    ffn_output = self.ffn(nout2, is_training=is_training)
    ffn_output = self.res_dropout3(ffn_output, training=is_training)
    out3 = ffn_output + out2  # (batch_size, target_seq_len, d_model)

    return out3


class DecoderBlock(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, inner_dropout, residual_dropout,
               attention_dropout, init):
    super(DecoderBlock, self).__init__()

    self.mha1 = att.MultiHeadAttention(d_model, num_heads, init)
    self.mha2 = att.MultiHeadAttention(d_model, num_heads, init)

    self.ffn = PointWiseFeedForwardNetwork(d_model, dff, inner_dropout, init)

    self.layernorm_cur = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm_pre = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm_com = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm_res = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.res_dropout1 = tf.keras.layers.Dropout(residual_dropout)
    self.res_dropout2 = tf.keras.layers.Dropout(residual_dropout)
    self.res_dropout3 = tf.keras.layers.Dropout(residual_dropout)

    self.attention_dropout = attention_dropout

  def call(self, inputs, **kwargs):
    cur_emb, pre_emb, enc_out = inputs
    is_training = kwargs["is_training"]
    look_ahead_mask = kwargs["look_ahead_mask"]
    padding_mask = kwargs["padding_mask"]
    dec_att_pen = kwargs["dec_att_pen"]
    enc_dec_att_pen = kwargs["enc_dec_att_pen"]

    # Block1: Decoder input
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    norm_cur_emb = self.layernorm_cur(cur_emb)
    if pre_emb is None:
      norm_pre_emb = norm_cur_emb
    else:
      norm_pre_emb = self.layernorm_pre(pre_emb)
    att_drop = self.attention_dropout if is_training else 0

    # (batch_size, target_seq_len, d_model)
    # input tuple: value, key, query
    attn1, atn_wgt_blk1 = self.mha1((norm_pre_emb, norm_pre_emb, norm_cur_emb),
                                    mask=look_ahead_mask,
                                    attention_dropout=att_drop,
                                    attention_penalty_mask=dec_att_pen)
    attn1 = self.res_dropout1(attn1, training=is_training)
    out1 = attn1 + cur_emb

    # Block2: Encoder to Decoder input
    nout1 = self.layernorm_com(out1)

    # (batch_size, target_seq_len, d_model)
    attn2, atn_wgt_blk2 = self.mha2((enc_out, enc_out, nout1),
                                    mask=padding_mask,
                                    attention_dropout=att_drop,
                                    attention_penalty_mask=enc_dec_att_pen)
    attn2 = self.res_dropout2(attn2, training=is_training)
    out2 = attn2 + out1  # (batch_size, target_seq_len, d_model)

    # Block3: Decoder output
    nout2 = self.layernorm_res(out2)

    # (batch_size, target_seq_len, d_model)
    ffn_output = self.ffn(nout2, is_training=is_training)
    ffn_output = self.res_dropout3(ffn_output, training=is_training)
    out3 = ffn_output + out2  # (batch_size, target_seq_len, d_model)

    return out3, atn_wgt_blk1, atn_wgt_blk2


def main():
  sample_encoder_layer = EncoderBlock(512, 8, 2048, 0.1, 0.1, 0.1,
                                      "glorot_uniform")

  sample_encoder_layer_output = sample_encoder_layer(
      tf.random.uniform((64, 43, 512)), False, None, None)

  print("A encoder layer output shape")
  # (batch_size, input_seq_len, d_model)
  print(sample_encoder_layer_output.shape)

  sample_decoder_layer = DecoderBlock(512, 8, 2048, 0.1, 0.1, 0.1,
                                      "glorot_uniform")

  sample_decoder_layer_output, _, _ = sample_decoder_layer(
      tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
      False, None, None, None)

  print("A decoder output shape")
  print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len,
  # d_model)

if __name__ == "__main__":
  main()
