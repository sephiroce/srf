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

# pylint: disable=import-error, too-few-public-methods, unused-argument
# pylint: disable=too-many-arguments

"""
attention.py: (multi-head)attention initiated from tensorflow tutorial.
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import numpy as np
import tensorflow as tf
from tfsr.helper.misc_helper import Util
import tfsr.helper.model_helper as mh

################################
# Scaled dot product attention #
################################
def scaled_dot_product_attention(query, key, value, mask, attention_dropout,
                                 att_pen_mask, is_debug=False):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    :param query: query shape == (..., seq_len_q, depth)
    :param key: key shape == (..., seq_len_k, depth)
    :param value: value shape == (..., seq_len_v, depth_v)
    :param mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
    :param att_pen_mask:
    :param attention_dropout:
    :param is_debug: to draw attention map

  Returns:
    output, attention_weights

  Equation:
    Attention(Q, K, V) = softmax ( ( Q * trans(K) ) / sqrt(d_k) ) * V

  Attention Penalty (from Speech Transformer paper):
    In addition, we encouraged the model attending to closer positions by adding
    bigger penalty on the attention weights of more distant position-pairs.

    there is no more specific description about attention penalty.
    this is my imagination,
    adding negative value for non-diagonal element to scaled_dotpd
    except for the first multi-head attention in decoders.

  """

  # Q * trans(K): (..., seq_len_q, seq_len_k)
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scaled matmul_qk: ( Q * trans(K) ) / sqrt(d_k)
  # d/h -> dimension / number of head?
  dimension_of_key = tf.cast(tf.shape(key)[-1], tf.float32) # <= d/h
  scaled_dotpd = matmul_qk / tf.math.sqrt(dimension_of_key)

  # [Batch, Headers, Feat seq, Lab seq]
  # Adding penalty to linearly normalized scaled_dotpd before masking it.
  if att_pen_mask is not None:
    scaled_dotpd += tf.math.log(1 + att_pen_mask) * -1
    if is_debug:
      tf.print(tf.shape(scaled_dotpd))

  # add the mask to the scaled tensor
  if mask is not None:
    scaled_dotpd += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1. (..., seq_len_q, seq_len_k)
  attention_weights = tf.nn.softmax(scaled_dotpd, axis=-1)

  if is_debug:
    Util.plot_attention_weights(attention_weights, None, "attention map")

  # dropout was added in front of a linear projection
  if 0 < attention_dropout < 1:
    attention_weights = tf.nn.dropout(attention_weights,
                                      rate=attention_dropout)

  output = tf.matmul(attention_weights, value)  # (..., seq_len_q, depth_v)

  return output, attention_weights

########################
# Multi-head attention #
########################
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, init):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert self.d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.dense_layer_for_query = tf.keras.layers.Dense(d_model, use_bias=False,
                                                       kernel_initializer=mh.get_init(init))

    self.dense_layer_for_key = tf.keras.layers.Dense(d_model, use_bias=False,
                                                     kernel_initializer=mh.get_init(init))

    self.dense_layer_for_value = tf.keras.layers.Dense(d_model, use_bias=False,
                                                       kernel_initializer=mh.get_init(init))

    self.dense = tf.keras.layers.Dense(d_model, use_bias=True,
                                       kernel_initializer=mh.get_init(init))

  def split_heads(self, input_vector, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len,
    depth)
    """
    input_vector = tf.reshape(input_vector, (batch_size, -1, self.num_heads,
                                             self.depth))
    return tf.transpose(input_vector, perm=[0, 2, 1, 3])

  def call(self, inputs, **kwargs):
    value, key, query = inputs
    mask = kwargs["mask"]
    att_drop = kwargs["attention_dropout"]
    att_pen = kwargs["attention_penalty_mask"]
    mha_batch_size = tf.shape(query)[0]

    query = self.dense_layer_for_query(query)  # (batch_size, seq_len, d_model)
    key = self.dense_layer_for_key(key)  # (batch_size, seq_len, d_model)
    value = self.dense_layer_for_value(value)  # (batch_size, seq_len, d_model)

    # (batch_size, num_heads, seq_len_q, depth)
    query = self.split_heads(query, mha_batch_size)
    # (batch_size, num_heads, seq_len_k, depth)
    key = self.split_heads(key, mha_batch_size)
    # (batch_size, num_heads, seq_len_v, depth)
    value = self.split_heads(value, mha_batch_size)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(query,
                                                                       key,
                                                                       value,
                                                                       mask,
                                                                       att_drop,
                                                                       att_pen)

    # (batch_size, seq_len_q, num_heads, depth)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # (batch_size, seq_len_q, d_model)
    concat_attention = tf.reshape(scaled_attention, (mha_batch_size, -1,
                                                     self.d_model))

    mha_output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

    return mha_output, attention_weights

def main():
  # Scaled dot product attention
  def print_out(query, key, value):
    temp_out, temp_attn = \
      scaled_dot_product_attention(query, key, value, None, 0, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

  np.set_printoptions(suppress=True)
  temp_k = tf.constant([[10, 0, 0],
                        [0, 10, 0],
                        [0, 0, 10],
                        [0, 0, 10]], dtype=tf.float32)  # (4, 3)

  temp_v = tf.constant([[1, 0],
                        [10, 0],
                        [100, 5],
                        [1000, 6]], dtype=tf.float32)  # (4, 2)

  # This `query` aligns with the second `key`,
  # so the second `value` is returned.
  temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
  print_out(temp_q, temp_k, temp_v)

  # This query aligns with a repeated key (third and fourth),
  # so all associated values get averaged.
  temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
  print_out(temp_q, temp_k, temp_v)

  # This query aligns equally with the first and second key,
  # so their values get averaged.
  temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
  print_out(temp_q, temp_k, temp_v)

  temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]],
                       dtype=tf.float32)  # (3, 3)
  print_out(temp_q, temp_k, temp_v)

  # Multi-head Attention
  temp_mha = MultiHeadAttention(d_model=512, num_heads=8, init="glorot_uniform")
  # (batch_size, encoder_sequence, d_model)
  input_vector = tf.random.uniform((1, 60, 512))

  # value, key, query
  out, attn = temp_mha((input_vector, input_vector, input_vector),
                       mask=None,
                       attention_dropout=0.0,
                       attention_penalty_mask=None)

  print("Multi Head Attention's Output shape")
  print(out.shape)
  print("Multi Head Attention's Attention shape")
  print(attn.shape)

if __name__ == "__main__":
  main()
