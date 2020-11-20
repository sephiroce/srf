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

# pylint: disable=import-error, too-few-public-methods,
# pylint: disable=pointless-string-statement

"""
train_helper.py: optimizer static methods
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  #pylint: disable=too-many-arguments
  def get_config(self):
    return {
        "model_dimension": self.d_model,
        "train_lr_param_k": self.train_lr_param_k,
        "warmup_steps": self.warmup_steps
    }

  def __init__(self, train_lr_param_k, d_model, warmup_steps,
               max_lr=10, dtype=None):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.dtype = tf.float32 if dtype is None else dtype
    self.d_model = tf.cast(self.d_model, self.dtype)
    self.train_lr_param_k = train_lr_param_k
    self.warmup_steps = warmup_steps
    self.max_lr = max_lr


  def __call__(self, step):
    arg1 = tf.cast(tf.math.rsqrt(step), self.dtype)
    arg2 = tf.cast(step * (self.warmup_steps ** -1.5), self.dtype)
    return tf.math.minimum(self.train_lr_param_k * tf.math.rsqrt(self.d_model)
                           * tf.math.minimum(arg1, arg2), self.max_lr)


def get_optimizer(config, dtype=None):
  if config.train_opti_type is None or \
    config.train_opti_type not in ["adam", "sgd"]:
    return tf.keras.optimizers.Adam(CustomSchedule(config.train_lr_param_k,
                                                   config.model_dimension,
                                                   config.train_warmup_n,
                                                   config.train_lr_max,
                                                   dtype),
                                    beta_1=config.train_adam_beta1,
                                    beta_2=config.train_adam_beta2,
                                    epsilon=config.train_adam_epsilon)
  if config.train_opti_type == "adam":
    return tf.keras.optimizers.Adam(learning_rate=config.train_lr_param_k)
  if config.train_opti_type == "sgd":
    return tf.keras.optimizers.SGD(learning_rate=config.train_lr_param_k)
  return None


def loss_ce(smoothing_type, labels, logits, confidence, output_dim):
  from tfsr.helper.common_helper import Constants
  if smoothing_type == Constants.SM_NEIGHBOR:
    return _loss_sm_neighbor(labels, logits, confidence, output_dim)
  if smoothing_type == Constants.SM_LABEL:
    return _loss_sm_label(labels, logits, confidence, output_dim)
  return None


def _loss_sm_neighbor(labels, logits, confidence, output_dim):
  # real [batch, label_length] to one hot vector [batch, label_length, vocab]
  ex_real = tf.one_hot(labels, depth=output_dim)

  if 0.0 < confidence < 1.0:
    """
    In order to prevent over-fitting, we used the neighborhood smoothing scheme
    proposed in Jan 2016, and the probability of correct label was set to 0.8.
    * Jan Chorowski and Navdeep Jaitly, “Towards better decoding and language
    model integration in sequence to sequence models,” arXiv preprint arXiv:1612
    .02695, 2016
    """
    pad_for_left = tf.constant([[0, 0, ], [0, 1], [0, 0]])
    pad_for_right = tf.constant([[0, 0, ], [1, 0], [0, 0]])
    ex_real = \
      (ex_real * confidence) + \
      tf.pad(ex_real[:, 1:, :], pad_for_left) * ((1 - confidence) / 2) + \
      tf.pad(ex_real[:, :-1, :], pad_for_right) * ((1 - confidence) / 2)

  loss_object = \
    tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
  loss_ = loss_object(ex_real, logits)

  mask = tf.math.logical_not(tf.math.equal(labels, 0))
  masked_loss = loss_ * tf.cast(mask, dtype=loss_.dtype)
  return masked_loss


def _loss_sm_label(labels, logits, confidence, output_dim):
  """Calculate cross entropy loss while ignoring padding.
    I referred to a padded_cross_entropy function of
  https://github.com/tensorflow/models/blob/master/official/nlp/transformer
  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    output_dim: int size of the vocabulary
  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
  with tf.name_scope("loss"):
    # Calculate smoothing cross entropy
    with tf.name_scope("smoothing_cross_entropy"):
      low_confidence = (1.0 - confidence) / tf.cast(output_dim - 1, tf.float32)
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=output_dim,
          on_value=confidence,
          off_value=low_confidence)
      xentropy = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=soft_targets)

      # Calculate the best (lowest) possible value of cross entropy, and
      # subtract from the cross entropy loss.
      norm_const = -(confidence * tf.math.log(confidence) +
                     tf.cast(output_dim - 1, tf.float32) *
                     low_confidence * tf.math.log(low_confidence + 1e-20))
      xentropy -= norm_const

    return xentropy * tf.cast(tf.not_equal(labels, 0), tf.float32)


def loss_function_w2v(real, pred, weights, smoothing=0):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                   reduction=tf.keras.losses.Reduction.NONE,
                                                   label_smoothing=smoothing)
  loss_ = loss_object(real, pred)
  loss_ = loss_ * weights

  return tf.reduce_sum(loss_)


def shuffle_data(texts):
  """
  Shuffle the data (called after making a complete pass through
  training or validation data during the training process)
  Params:
    texts (list): Sentences uttered in each audio clip
  """
  perm = np.random.permutation(len(texts))
  texts = [texts[i] for i in perm]
  return texts


def ppl(labels, logits, seq_len):
  # calculates an batch-wise accumulated log probability
  full_logprob = \
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels= \
                                                     tf.cast(labels, tf.int32),
                                                   logits=logits)

  # generating sequence mask to handle variable length of inputs
  # in this case actually squeeze is not needed! but my lm toolkit
  # needs it. (I'll figure it out someday later..)
  seq_mask = tf.squeeze(tf.sequence_mask(seq_len,
                                         maxlen= \
                                           tf.shape(full_logprob)[1],
                                         dtype=tf.float32))

  logprob = tf.reduce_sum(full_logprob * seq_mask)

  # calculates ppl
  return logprob


def loss_ewerr(hyposs, labels, lprobss, vocab, is_debug=False):
  """
  Rohit et al, MINIMUM WORD ERROR RATE TRAINING FOR ATTENTION-BASED
  SEQUENCE-TO-SEQUENCE MODELS, ICASSP 2018

  Loss_nbest_werrs = sum_yi.IN.Beam (P_hat(yi | X) * [WE(y_i, y*) - W_hat])
  The three terms in the right side of the equation above.
  1) P_hat(yi | X) = P(yi|X) / sum_yi.IN.Beam [P(yi|X)]
  the distribution re-normalized over just the N-best hypotheses.

  2) WE(y_i, y*) = edit_distance(y_i, y*)
  the number of word errors

  3) W_hat = 1/BeamN * sum_yi.IN.Beam[WER(y_i, y*)]
  W_hat is the average number of word errors over the N-best hypohtheses
  Average WERs over beams [batch, beam].
  "average number of word errors" -> "average of word error rates", because
  In section 3.1, W_hat = 1/N * sum_0toN[WER(y_i, y*)] is the average number
  of word errors over the samples.

  :param hyposs: [batch, beam, tar_len]
  :param labels: [batch, tar_len]
  :param lprobss: [batch, beam]
  :param vocab:
  :param is_debug:
  :return:
  """
  # pylint: disable=too-many-locals
  batch_size = tf.shape(hyposs)[0]
  beam_width = tf.shape(hyposs)[1]

  if is_debug:
    tf.print("shape of hyps [batch, beam, tar_len]", tf.shape(hyposs))
    tf.print("shape of labs [batch, tar_len]", tf.shape(labels))
    tf.print("shape of lprs [batch, beam]", tf.shape(lprobss))

  # 1) P_hat(yi | X) = P(yi|X) / sum_yi.IN.Beam [P(yi|X)]
  # [batch, beam]
  probss = tf.exp(lprobss)
  inv_probss_sum = 1 / tf.reduce_sum(probss, -1)
  p_hat = tf.einsum("ij, i -> ij", probss, inv_probss_sum)

  # 2) Word errors: [batch, beam]
  flat_hyps = tf.reshape(hyposs, [batch_size * beam_width, -1])
  flat_labs = tf.reshape(tf.tile(tf.expand_dims(labels, 1), [1, beam_width, 1]),
                         [batch_size * beam_width, -1])
  if is_debug:
    tf.print("shape of flat hyps [batch * beam, tar_len]", tf.shape(flat_hyps))
    tf.print("shape of flat labs [batch * beam, tar_len]", tf.shape(flat_labs))

  flat_errs, _ = compute_wer(tf.cast(flat_hyps, tf.int32),
                             tf.cast(flat_labs, tf.int32), vocab)

  if is_debug:
    tf.print("shape of flat wers [batch * beam]", tf.shape(flat_errs))

  errs = tf.reshape(flat_errs, [batch_size, beam_width])

  # 3) W_hat = 1/BeamN * sum_yi.IN.Beam[WER(y_i, y*)]
  w_hat = tf.reduce_sum(errs, -1) / tf.cast(beam_width, tf.float32)
  # expand w_hat to be broadcastable
  w_hat = tf.tile(tf.expand_dims(w_hat, 1), [1, beam_width])

  # Loss_nbest_werrs = sum_yi.IN.Beam (P_hat(yi | X) * [WER(y_i, y*) - W_hat])
  werrs = tf.reduce_sum(tf.math.multiply(p_hat, errs - w_hat), -1)

  if is_debug:
    tf.print("phat:", p_hat, summarize=100)
    tf.print("errs:", errs, summarize=100)
    tf.print("what:", w_hat, summarize=100)
    tf.print("1) shape of Phat [batch, beam]", tf.shape(p_hat))
    tf.print("2) shape of ERRs [batch, beam]", tf.shape(errs))
    tf.print("3) shape of What [batch, beam]", tf.shape(w_hat))
    tf.print("4) shape of Expected word error numbers [batch]", tf.shape(werrs))

  return werrs

def get_bucket_info(batch_total_size, num_gpus, min_bkt, max_bkt, step,
                    step_for_bucket_size=False, manual_bucket_batch_sizes=None):
  #pylint: disable=too-many-arguments, too-many-locals
  """
  :param manual_bucket_batch_sizes:
  :param batch_total_size: a total length in a batch
  :param num_gpus:
  :param min_bkt:
  :param max_bkt:
  :param step:
  :param step_for_bucket_size:
  :return:
  """
  bucket_boundaries = []
  bucket_batch_sizes = []
  if step_for_bucket_size and manual_bucket_batch_sizes is None:
    max_buckets = int(np.floor(batch_total_size / min_bkt))
    for batch_size in range(max_buckets, num_gpus, -step):
      # batch_size * boundary = batch_total_size
      boundary = int(np.floor(batch_total_size / batch_size))
      if batch_size > num_gpus:
        bucket_batch_sizes.append(batch_size)
      else:
        break
      bucket_boundaries.append(boundary if boundary < max_bkt else max_bkt)
      if boundary >= max_bkt:
        break
    bucket_batch_sizes.append(num_gpus)
  else:
    boundaries = manual_bucket_batch_sizes if manual_bucket_batch_sizes else\
      range(min_bkt, max_bkt + step, step)

    for boundary in boundaries:
      # batch_size * boundary = batch_total_size
      batch_size = int(np.floor(batch_total_size / boundary))
      if batch_size > num_gpus:
        bucket_batch_sizes.append(batch_size)
      else:
        break
      bucket_boundaries.append(boundary)
    bucket_batch_sizes.append(num_gpus)

  # removing duplicated sizes
  prev = -1
  length = len(bucket_boundaries)
  for i in reversed(range(length)):
    if bucket_batch_sizes[i] == prev:
      bucket_boundaries.pop(i)
      bucket_batch_sizes.pop(i)
    prev = bucket_batch_sizes[i]

  return bucket_boundaries, bucket_batch_sizes


def compute_wer(hyp, ref, vocab, is_debug=False):
  """
  # initiated from https://github.com/tensorflow/tensor2tensor/pull/1242/files

  :param hyp:
  :param ref:
  :param vocab:
    A list mapping indices to output tokens.
  :param is_debug
  :return:
    The word error rate.
  """

  def assemble_to_words(subwords):
    # pylint: disable=anomalous-backslash-in-string
    gathered = tf.gather(vocab, tf.cast(subwords, tf.int32))
    joined = tf.strings.regex_replace(tf.strings.reduce_join(gathered, axis=1),
                                      b'<EOS>.*', b'')
    cleaned = tf.strings.regex_replace(joined, b'n', b'') # non-lang syms
    cleaned = tf.strings.regex_replace(cleaned, b'@ ', b'') # bos <space>
    cleaned = tf.strings.regex_replace(cleaned, b' \$', b'') # <space> eos
    cleaned = tf.strings.regex_replace(cleaned, b'\$', b'') # eos
    cleaned = tf.strings.regex_replace(cleaned, b'@', b'') # bos
    cleaned = tf.strings.regex_replace(cleaned, b'p', b'') # padding syms
    cleaned = tf.strings.regex_replace(cleaned, b'@@ ', b'') # bpe
    cleaned = tf.strings.regex_replace(cleaned, b' +', b' ') # double blanks
    cleaned = tf.strings.regex_replace(cleaned, b'^ ', b'') # strip
    cleaned = tf.strings.regex_replace(cleaned, b' $', b'') # strip
    tokens = tf.compat.v1.string_split(cleaned, ' ')
    return tokens

  # expanding to appropriate dimensions
  max_value = tf.reduce_max([tf.reduce_max(ref), tf.reduce_max(hyp)]) + 1
  targets = tf.expand_dims(ref, -1)
  targets = tf.expand_dims(targets, -1)

  y_preds = tf.one_hot(hyp, max_value, on_value=1, off_value=0, dtype=tf.int32)
  y_preds = tf.expand_dims(y_preds, 2)
  y_preds = tf.expand_dims(y_preds, 2)

  # get sparse tensor
  y_preds = tf.squeeze(tf.argmax(y_preds, axis=-1), axis=(2, 3))
  targets = tf.squeeze(targets, axis=(2, 3))
  reference = assemble_to_words(targets)
  predictions = assemble_to_words(y_preds)

  if is_debug:
    tf.print(reference, summarize=10000)
    tf.print(predictions, summarize=10000)

  d_ref = tf.sparse.to_dense(reference)
  ref_lens = tf.cast(tf.math.count_nonzero(d_ref, axis=-1), tf.int32)
  distances = tf.edit_distance(predictions, reference, normalize=False)
  reference_length = tf.size(reference.values, out_type=tf.int32)
  tf.debugging.assert_equal(tf.reduce_sum(ref_lens), reference_length)

  return distances, tf.cast(ref_lens, tf.float32)


def prep_process(labels, feat_len, tar_len, feats, in_len_div):
  # Cropping input
  max_feat_len = tf.math.reduce_max(feat_len, keepdims=False)
  feats = feats[:, :max_feat_len, :]
  max_tar_len = tf.math.reduce_max(tar_len, keepdims=False)

  import tfsr.helper.model_helper as mh
  enc_pad_mask = mh.get_padding_bias(feat_len, in_len_div)

  if labels is None:
    return feats, enc_pad_mask

  labels = labels[:, :max_tar_len]

  # @ a b c $
  tar_inp = labels[:, :-1]  # @ a b c
  tar_real = labels[:, 1:]  # a b c $
  comb_mask = mh.create_combined_mask(tar_inp)

  return feats, tar_inp, tar_real, enc_pad_mask, comb_mask


def main():
  temp_learning_rate_schedule = CustomSchedule(train_lr_param_k=10,
                                               d_model=256,
                                               warmup_steps=36111)

  plt.plot(temp_learning_rate_schedule(tf.range(144445, dtype=tf.float32)))
  plt.ylabel("Learning Rate")
  plt.xlabel("Train Step")
  plt.show()

  bucket_boundaries, bucket_batches = get_bucket_info(20000, 2, 200, 1000, 100)
  print(bucket_boundaries)
  print(bucket_batches)

  ## WER
  print("\nTest for Word Error Rate")
  from tfsr.helper.misc_helper import Util
  vocab, _, _, _ = Util.load_vocab("samples/data/stf.vocab")

  y_preds = tf.constant([[29, 20, 8, 5, 29, 8, 9, 19, 20, 15, 18, 25, 29, 15,
                          6, 29, 15, 20, 8, 5, 18, 29, 1, 6, 18, 9, 3, 1, 14,
                          29, 14, 1, 20, 9, 15, 14, 19, 29, 9, 19, 29, 14, 25,
                          29, 7, 21, 9, 4, 5, 29, 23, 8, 9, 20, 5, 19, 29, 9,
                          14, 4, 9, 1, 14, 19, 29, 1, 14, 4, 29, 19, 13, 1, 12,
                          12, 29, 2, 12, 1, 3, 11, 29, 20, 18, 9, 5, 29, 19,
                          8, 15, 21, 12, 4, 29, 6, 5, 1, 18, 29, 31, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0],
                         [21, 20, 21, 18, 5, 19, 29, 23, 5, 18, 5, 29, 19, 21,
                          16, 16, 15, 18, 20, 5, 4, 29, 2, 25, 29, 3, 15, 14, 3,
                          5, 18, 14, 29, 20, 8, 1, 20, 29, 23, 8, 5, 1, 20, 29,
                          13, 1, 25, 29, 2, 5, 29, 22, 15, 12, 21, 14, 5, 18, 1,
                          2, 12, 5, 29, 9, 6, 29, 3, 15, 12, 4, 29, 19, 14, 1,
                          16, 19, 29, 22, 15, 12, 1, 20, 9, 19, 29, 16, 1, 19,
                          20, 29, 23, 5, 5, 11, 5, 14, 4, 19, 29, 23, 1, 18, 14,
                          20, 8, 29, 1, 14, 1, 12, 25, 19, 20, 19, 29, 19, 1, 9,
                          4, 29, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

  targets = tf.constant([[6, 29, 20, 8, 5, 29, 8, 9, 19, 20, 15, 18, 25, 29, 15,
                          6, 29, 15, 20, 8, 5, 18, 29, 1, 6, 18, 9, 3, 1, 14,
                          29, 14, 1, 20, 9, 15, 14, 19, 29, 9, 19, 29, 1, 14,
                          25, 29, 7, 21, 9, 4, 5, 29, 23, 8, 9, 20, 5, 19, 29,
                          9, 14, 4, 9, 1, 14, 19, 29, 1, 14, 4, 29, 19, 13, 1,
                          12, 12, 29, 2, 12, 1, 3, 11, 29, 20, 18, 9, 5, 29, 19,
                          8, 15, 21, 12, 4, 29, 6, 5, 1, 18, 29, 31, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [6, 21, 20, 21, 18, 5, 19, 29, 23, 18, 5, 29, 19, 21,
                          16, 16, 15, 18, 20, 5, 4, 29, 2, 25, 29, 3, 15, 14, 3,
                          5, 18, 14, 29, 20, 8, 1, 20, 29, 23, 8, 5, 1, 20, 29,
                          13, 1, 25, 29, 2, 5, 29, 22, 15, 12, 21, 14, 5, 18, 1,
                          2, 12, 5, 29, 9, 6, 29, 3, 15, 12, 4, 29, 19, 14, 1,
                          16, 19, 29, 22, 15, 12, 1, 20, 9, 19, 29, 16, 1, 19,
                          20, 29, 23, 5, 5, 11, 5, 14, 4, 19, 29, 23, 1, 18, 14,
                          20, 8, 29, 1, 14, 1, 12, 25, 19, 20, 19, 29, 19, 1, 9,
                          4, 29, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

  places = 10 ** 3
  actual_errs, actual_words = compute_wer(y_preds, targets, vocab, True)
  actual_wers = tf.math.round((actual_errs / actual_words) * places) / places
  expected_wers = tf.constant([0.111, 0.105])
  print(actual_errs)
  print(actual_words)
  print(actual_wers)
  print(expected_wers)

  tf.debugging.assert_equal(actual_wers, expected_wers)

if __name__ == "__main__":
  main()
