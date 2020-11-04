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
import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def get_config(self):
    return {
        "model_dimension": self.d_model,
        "train_lr_param_k": self.train_lr_param_k,
        "warmup_steps": self.warmup_steps
    }

  # pylint: disable=too-many-arguments
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
  return tf.keras.optimizers.Adam(CustomSchedule(config.train_lr_param_k,
                                                 config.model_dimension,
                                                 config.train_warmup_n,
                                                 config.train_lr_max,
                                                 dtype),
                                  beta_1=config.train_adam_beta1,
                                  beta_2=config.train_adam_beta2,
                                  epsilon=config.train_adam_epsilon)


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
