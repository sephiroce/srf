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

# pylint: disable=import-error, too-many-locals, too-many-arguments
# pylint: disable=too-many-branches, too-many-statements, no-member

import tensorflow as tf

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

num_parallel_calls = tf.data.experimental.AUTOTUNE

def create_ds(file_pattern, shuffle, max_inp, max_tar, is_utt_id=False):
  """Create dataset where each item is a dict of "inputs" and "targets".
    :arg file_pattern: String used to match the input TFRecord files.
    :arg batch_size: Maximum number of tokens per global batch of examples.
    :arg max_length: Maximum number of tokens per example
    :arg num_parallel_calls: Number of cpu cores for parallel input processing.
    :arg shuffle: If true, randomizes order of elements.
    :arg repeat: Number of times to repeat the dataset. If None, the dataset is
      repeated forever.
    :arg num_replicas: Number of GPUs or other workers. We will generate global
      batches, and each global batch is equally divisible by number of replicas.
      Currently it is only effective when static_batch==True.
      TODO: make it effective when static_batch=False.
  Returns:
    tf.data.Dataset object containing examples loaded from the files.
  """

  def _load_records(filename):
    """Read file and return a dataset of tf.Examples."""
    return tf.data.TFRecordDataset(filename, buffer_size=100 * 1000 * 1000,
                                   num_parallel_reads=10)

  def _filter_max_length(example, max_length, idx):
    """max_length in seconds, sampled per 10 ms, so divide it by 100"""
    return tf.logical_or(max_length < 1, example[idx] <= max_length)

  def _parse_example(serialized_example):
    """Return inputs and targets Tensors from a serialized tf.Example."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "input_speech": tf.io.VarLenFeature(tf.float32),
            "target_label": tf.io.VarLenFeature(tf.int64),
            "input_length": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "target_length": tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        }
    )
    inputs = tf.sparse.to_dense(features["input_speech"])
    targets = tf.sparse.to_dense(features["target_label"])
    input_length = features["input_length"]
    target_length = features["target_length"]

    return inputs, targets, input_length, target_length

  def _parse_example_utt_id(serialized_example):
    """Return inputs and targets Tensors from a serialized tf.Example."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "input_speech": tf.io.VarLenFeature(tf.float32),
            "target_label": tf.io.VarLenFeature(tf.int64),
            "input_length": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "target_length": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "utt_id": tf.io.FixedLenFeature(shape=(), dtype=tf.string)
        }
    )
    inputs = tf.sparse.to_dense(features["input_speech"])
    targets = tf.sparse.to_dense(features["target_label"])
    input_length = features["input_length"]
    target_length = features["target_length"]
    utt_id = features["utt_id"]

    return inputs, targets, input_length, target_length, utt_id

  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  # Read files and interleave results. When training, the order of the examples
  # will be non-deterministic.
  dataset = dataset.interleave(_load_records, cycle_length=None,
                               num_parallel_calls=num_parallel_calls,
                               deterministic=False)

  # Parse each tf.Example into a dictionary
  if is_utt_id:
    dataset = dataset.map(_parse_example_utt_id,
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.filter(
        lambda x, y, a, b, c: _filter_max_length((x, y, a, b, c), max_inp, 2))
    dataset = dataset.filter(
        lambda x, y, a, b, c: _filter_max_length((x, y, a, b, c), max_tar, 3))
  else:
    # TODO: Look into prefetch_input_elements for performance optimization.
    dataset = dataset.map(_parse_example, num_parallel_calls=num_parallel_calls)
    dataset = dataset.filter(
        lambda x, y, a, b: _filter_max_length((x, y, a, b), max_inp, 2))
    dataset = dataset.filter(
        lambda x, y, a, b: _filter_max_length((x, y, a, b), max_tar, 3))

  return dataset


def finalize_ds(dataset, repeat, shuffle):
  dataset = dataset.cache()
  if shuffle:
    dataset = dataset.shuffle(buffer_size=5000)
  dataset = dataset.repeat(repeat)

  # Prefetch the next element to improve speed of input pipeline.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


def create_ds_batch_for_test(file_pattern, batch_size, max_inp, max_tar):
  # Sanity check since this toolkit not support drop_remainder=True yet.
  import glob
  utt_num = 0
  for file_name in glob.glob(file_pattern):
    utt_num += sum(1 for _ in tf.data.TFRecordDataset(file_name))
  # TODO: A hard coded batch size for a test data set.
  if utt_num % batch_size != 0:
    batch_size = 1 # Hard coded!!

  dataset = create_ds(file_pattern, False, max_inp, max_tar, True)
  dataset = dataset.padded_batch(
      # First calculate batch size (token number) per worker, then divide it
      # into sentences, and finally expand to a global batch. It could prove
      # the global batch divisible for distribution strategy.
      batch_size=batch_size,
      padded_shapes=([None], [None], [], [], []), drop_remainder=False)

  return finalize_ds(dataset, 1, False)


def create_ds_batch_for_train(file_pattern, shuffle, repeat,
                              batch_size, max_inp, max_tar):
  dataset = create_ds(file_pattern, shuffle, max_inp, max_tar)
  dataset = dataset.padded_batch(
      # First calculate batch size (token number) per worker, then divide it
      # into sentences, and finally expand to a global batch. It could prove
      # the global batch divisble for distribution strategy.
      batch_size=batch_size,
      padded_shapes=([None], [None], [], []), drop_remainder=True)

  return finalize_ds(dataset, repeat, shuffle)


def create_ds_bucket(file_pattern, shuffle, repeat,
                     bucket_boundaries, bucket_batch_sizes, max_inp, max_tar):

  def element_length_fn(inputs, targets, input_length, target_length):
    # pylint: disable=unused-argument
    return tf.cast(input_length, tf.int32)

  dataset = create_ds(file_pattern, shuffle, max_inp, max_tar)
  dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
      element_length_func=element_length_fn,
      bucket_boundaries=bucket_boundaries,
      bucket_batch_sizes=bucket_batch_sizes,
      pad_to_bucket_boundary=False,
      no_padding=False, drop_remainder=True))
  return finalize_ds(dataset, repeat, shuffle)


def map_data_for_transformer_fn(inputs, targets, input_length, target_length,
                                arg):
  """ it reshapes 1D padded features to 2D.
  :param inputs: inputs
  :param targets: targets
  :param input_length: input_length
  :param target_length: target_length
  :param arg: feature_dimension
  :return:
  """

  return tf.reshape(inputs, [tf.shape(inputs)[0], -1, arg]), \
         tf.cast(targets, tf.int32), \
         tf.cast(input_length, tf.int32), \
         tf.cast(target_length, tf.int32)


def map_data_for_transformer_utt_id_fn(inputs, targets, input_length,
                                       target_length, utt_id, arg):
  """ it reshapes 1D padded features to 2D.
  :param inputs: inputs
  :param targets: targets
  :param input_length: input_length
  :param target_length: target_length
  :param arg: feature_dimension
  :return:
  """

  return tf.reshape(inputs, [tf.shape(inputs)[0], -1, arg]), \
         tf.cast(targets, tf.int32), \
         tf.cast(input_length, tf.int32), \
         tf.cast(target_length, tf.int32), \
         tf.cast(utt_id, tf.string)
