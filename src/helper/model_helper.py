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

"""model_helper.py: methods for models"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import math
import tensorflow as tf
import matplotlib.pylab as plt
from tfsr.helper.common_helper import Constants

#######################
# Positional Encoding #
#######################
def get_pos_enc(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.
  I borrowed from the official transformer model.
  URL: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulated in Attention is All You Need, section 3.5.
  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
  Returns:
    Tensor with shape [length, hidden_size]
  """
  # We compute the positional encoding in float32 even if the model uses
  # float16, as many of the ops used, like log and exp, are numerically unstable
  # in float16.
  position = tf.cast(tf.range(length), tf.float32)
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales, tf.float32) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  return signal


###########
# Masking #
###########
def create_padding_mask(seq):
  """
  Mask all the pad tokens in the batch of sequence. It ensures that the model
  does not treat padding as the input. The mask indicates where pad value 0
  is present: it outputs a 1 at those locations, and a 0 otherwise.

  seq: a sequence padded with zeros
  """
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def get_padding_bias(inp_len, strides=4):
  """Calculate bias tensor from padding values in tensor.

  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length].
  The tensor is zero at non-padding locations, and -1e9 (negative infinity)
  at padding locations.

  Args:
    inp_len: int tensor with shape [batch_size], represents input speech lengths
    strides: time domain strides * the number of cnn layers

  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, inp_len].

  """
  with tf.name_scope("attention_bias"):
    inp_len = tf.math.ceil(inp_len / strides)
    attention_bias = tf.abs(tf.sequence_mask(inp_len, dtype=tf.dtypes.float32) - 1.0)
  return attention_bias[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
  """
  The look-ahead mask is used to mask the future tokens in a sequence. In other
  words, the mask indicates which entries should not be used.

  This means that to predict the third word, only the first and second word will
  be used. Similarly to predict the fourth word, only the first, second and the
  third word will be used and so on.

  size: the length of label sequences
  """
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def create_combined_mask(tar):
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  return tf.maximum(dec_target_padding_mask, look_ahead_mask)


def feat_mask(args):
  """
  I do not know why it is not working!
  tf.cond(tf.rank(args[0]) == 4,
  lambda: tf.einsum('ijkl, ij->ijkl', args[0], mask),
  lambda: tf.einsum('ijk, ij->ijk', args[0], mask))

  [input_sequence, input_lengths]
  :param args:
  :return:
  """
  lengths = tf.math.ceil(tf.cast(args[1], tf.dtypes.int32) / args[2])
  mask = tf.sequence_mask(lengths, dtype=args[0].dtype)

  result = tf.einsum('ijkl, ij->ijkl', args[0], mask)
  return result


def feat_mask2(args):
  """
  [input_sequence, input_lengths]
  :param args:
  :return:
  """
  lengths = tf.math.ceil(tf.cast(args[1], tf.dtypes.int32) / args[2])
  mask = tf.sequence_mask(lengths, dtype=args[0].dtype)

  result = tf.einsum('ijk, ij->ijk', args[0], mask)
  return result


def get_init(init):
  if init == Constants.INIT_FANAVG:
    return tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                 distribution='uniform',
                                                 seed=None)
  return Constants.INIT_GLOROT


def get_decoder_self_attention_bias(length, dtype=tf.float32):
  """Calculate bias for decoder that maintains model's autoregressive property.
  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.
  Args:
    length: int length of sequences in batch.
    dtype: The dtype of the return value.
  Returns:
    float tensor of shape [1, 1, length, length]
  """
  neg_inf = -1e9
  with tf.name_scope("decoder_self_attention_bias"):
    valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype),
                                     -1, 0)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = neg_inf * (1.0 - valid_locs)
  return decoder_bias

#####################
# Attention penalty #
#####################
def create_attention_penalty(config, logger):
  #pylint: disable=too-many-boolean-expressions
  # Creating attention penalty
  if (config.model_ap_encoder or config.model_ap_decoder or config.model_ap_encdec) and \
      config.model_ap_width_zero is not None and config.model_ap_width_zero > 0 and \
      config.model_ap_width_stripe is not None and config.model_ap_width_stripe > 0 and \
      config.model_ap_scale is not None and config.model_ap_scale > 0.0:
    att_pen = AttentionPenalty(max_len=2500, # 100s= 100Kms = 2500 * 4 * 10 ms
                               num_head=config.model_att_head_num,
                               zero_width=config.model_ap_width_zero,
                               stripe_width=config.model_ap_width_stripe,
                               scale=config.model_ap_scale)
    logger.info("An attention penalty board was built with a zero width %d, "
                "a stripe width %d and a scale factor %f",
                config.model_ap_width_zero, config.model_ap_width_stripe,
                config.model_ap_scale)
    logger.info("Attention penalties mask will be applied for")
    if config.model_ap_encoder:
      logger.info("> Encoder self-attention")
    if config.model_ap_decoder:
      logger.info("> Decoder self-attention")
    if config.model_ap_encdec:
      logger.info("> Encoder-Decoder attention")
  else:
    att_pen = None
    logger.info("Attention penalties will not be applied.")

  return att_pen


class AttentionPenalty:
  #pylint: disable=too-many-arguments
  def __init__(self, max_len, num_head, zero_width, stripe_width, scale):
    att_penalty = tf.ones([max_len, max_len])
    self.eap = tf.zeros(([num_head, max_len, max_len]))
    for i in range(zero_width - 1, max_len, stripe_width):
      self.eap += tf.abs(1 - tf.linalg.band_part(att_penalty, i, i,
                                                 name="attention_penalty_board"))
    self.num_head = num_head
    self.max_len = max_len
    self.eap *= scale

  @property
  def big_penalty_map(self):
    return self.eap

  def create_pens(self, config, inp_len, tar_len=None):
    #pylint: disable=too-many-locals
    inp_len = tf.cast(inp_len, tf.int32)
    enc_max_len = tf.math.reduce_max(inp_len, keepdims=False)
    enc_att_pen = self.eap[:, :enc_max_len, :enc_max_len]

    enc_dec_att_pen, dec_att_pen, dec_max_len = None, None, None

    if tar_len is not None:
      dec_max_len = tf.cast(tf.math.reduce_max(tar_len, keepdims=False), tf.int32)

      if config.model_ap_encdec:
        enc_dec_att_pen = self.eap[ :, :dec_max_len, :enc_max_len]

      if config.model_ap_decoder:
        dec_att_pen = self.eap[ :, :dec_max_len, :dec_max_len]

    return enc_att_pen, enc_dec_att_pen, dec_att_pen


###################
# Model selection #
###################
def select_model(config, logger):
  #pylint: disable=import-outside-toplevel
  from tfsr.model.transformer import Transformer
  from tfsr.model.transformer_mf import TransformerMF
  from tfsr.model.transformer_fs import TransformerFS

  if config.feat_dim1 and config.feat_dim2:
    assert config.feat_dim1 + config.feat_dim2 == config.feat_dim
    Model = TransformerMF
    logger.info("Model: Transformer Multi Feature")
  elif config.model_type == "stf_fs":
    Model = TransformerFS
    logger.info("Model: Transformer Future Sequence")
  else:
    Model = Transformer
    logger.info("Model: Transformer")

  return Model


def main():
  #pylint: disable=too-many-statements
  # Testing Attention Penalty
  class Config:
    def __init__(self):
      self.model_ap_encdec = None
      self.model_ap_decoder = None
  config = Config()

  num_head = 4

  # Tiny case
  inp_len = tf.constant([10, 10])
  tar_len = tf.constant([8, 2])
  att_pen = AttentionPenalty(1500, num_head, 1, 1, 1.0)

  config.model_ap_encdec = True
  config.model_ap_decoder = True
  eap, edap, dap = att_pen.create_pens(config, inp_len, tar_len)

  plt.pcolormesh(eap[0], cmap='viridis')
  plt.gca().invert_yaxis()
  plt.show()
  print(eap[0])

  plt.pcolormesh(edap[0], cmap='viridis')
  plt.gca().invert_yaxis()
  plt.show()
  print(edap[0])

  plt.pcolormesh(edap[1], cmap='viridis')
  plt.gca().invert_yaxis()
  plt.show()
  print(edap[1])

  plt.pcolormesh(dap[0], cmap='viridis')
  plt.gca().invert_yaxis()
  plt.show()
  print(dap[0])

  plt.pcolormesh(dap[1], cmap='viridis')
  plt.gca().invert_yaxis()
  plt.show()
  print(dap[1])

  # Real case
  att_pen = AttentionPenalty(1500, num_head, 10, 20, 1.0)
  inp_len = tf.constant([366, 618, 618, 618, 618, 618, 618, 618, 618, 618, 618,
                         618, 618, 618, 618, 618, 618, 618, 618, 618, 618, 618])
  tar_len = tf.constant([200, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
                         150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150])

  eap, edap, dap = att_pen.create_pens(config, inp_len, tar_len)

  plt.pcolormesh(eap[0], cmap='viridis')
  plt.gca().invert_yaxis()
  plt.show()

  plt.pcolormesh(dap[0], cmap='viridis')
  plt.gca().invert_yaxis()
  plt.show()

  plt.pcolormesh(dap[1], cmap='viridis')
  plt.gca().invert_yaxis()
  plt.show()

  # Testing positional encoding
  print("Test positional encoding")
  model_dim = 256
  pos_encoding = get_pos_enc(1040, model_dim, min_timescale=1.0,
                             max_timescale=1.0e4)
  print(pos_encoding.shape)

  plt.pcolormesh(pos_encoding, cmap='RdBu')
  plt.xlabel('Depth')
  plt.xlim((0, model_dim))
  plt.ylabel('Position')
  plt.colorbar()
  plt.show()

  # Testing padding mask
  print("Test Padding mask")
  input_vector = tf.constant([[7, 6, 0, 0, 1],
                              [1, 2, 3, 0, 0],
                              [0, 0, 0, 4, 5]])

  tmp = create_padding_mask(input_vector)

  print(tmp)

  # Testing padding bias
  print("Test Padding bias")
  input_vector = tf.constant([18, 12, 16])
  tmp = get_padding_bias(input_vector)
  print(tmp)

  # Testing look ahead mask
  input_vector = tf.random.uniform((1, 10))
  temp = create_look_ahead_mask(input_vector.shape[1])
  print(temp)

  # Testing decoder self attention mask
  temp = get_decoder_self_attention_bias(10)
  print(temp)

if __name__ == "__main__":
  main()
