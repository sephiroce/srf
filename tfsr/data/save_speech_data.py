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

"""
Save WAV files and the corresponding transcriptions into TF records format.
This script was built with reference to
https://github.com/tensorflow/models/blob/master/official/transformer/data_download.py
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import os
import random
import sys
import json
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
from tfsr.helper.misc_helper import Util
from tfsr.helper.common_helper import Logger, ParseOption, ExitCode, Tag

def convert_to_tfrecord(logger, config, data_set, cmvn, is_debug=False):
  """
  :param is_debug:
  :param cmvn:
  :param config:
  :param data_set:
  :param logger:

  raw_files: $raw_files.adc: speech list, $raw_files.tra: transcription list
  tag: String that will be added onto the file names. (train, valid, eval)
  total_shards: Number of files to divide the data into.

  Returns:
    List of all files produced.

  """
  def _int64_list_feat(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _float_list_feat(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def _string_list_feat(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  data_path = config.path_base
  feat_type = config.feat_type
  feat_dim = config.feat_dim
  data_name = config.prep_data_name
  if config.path_wrt_tfrecord is None:
    logger.critical("path-wrt-tfrecord is None")
    sys.exit(1)
  tfrecord_dir = config.path_wrt_tfrecord
  is_char = config.prep_data_unit == 'char'

  Util.make_dir(Util.get_file_path(data_path, tfrecord_dir))

  if data_set == Tag.TRAIN:
    meta_file = Util.get_file_path(data_path, config.path_train_json)
    tf_record_name = data_set
    total_shards = config.prep_data_shard
  elif data_set == Tag.VALID:
    meta_file = Util.get_file_path(data_path, config.path_valid_json)
    tf_record_name = data_set
    total_shards = 1
  elif data_set == Tag.TEST:
    meta_file = Util.get_file_path(data_path, config.path_test_json)
    tf_record_name = data_set
    total_shards = 1
  else:
    logger.critical("type of data set must be the one among %s, %s, %s but "
                    " %s was provided.", Tag.TRAIN, Tag.VALID, Tag.TEST,
                    data_set)
    sys.exit(ExitCode.INVALID_OPTION)

  # Loading a vocabulary
  vocab_path = Util.get_file_path(data_path, config.path_vocab)
  if not os.path.isfile(vocab_path):
    logger.critical("%s does not exist.", vocab_path)
    sys.exit(ExitCode.INVALID_FILE_PATH)
  _, vocab, _, _ = Util.load_vocab(vocab_path, logger)

  # Create a file for each shard.
  Util.make_dir(data_path + "/" + tfrecord_dir)
  tfrecord_paths = []
  for shard_num in range(total_shards):
    tfrecord_path = \
      "%s-%s-%s-%d-%.5d-of-%.5d" % (data_name, tf_record_name, feat_type, feat_dim,
                                    shard_num + 1, total_shards)
    tfrecord_paths.append(os.path.join(data_path + "/" + tfrecord_dir,
                                       tfrecord_path))

  counter = 0
  # If already exists then just return the list of paths
  if Util.all_exist(tfrecord_paths):
    logger.info("TFRecords of %s already exist." % tfrecord_paths)
  else:
    logger.info("TFRecords of %s are being saved into %s"%(meta_file,
                                                           tfrecord_paths))
    current_milli_time = Util.currentTimeMillis()

    # Write examples to each shard in round robin order.
    tmp_file_paths = [fname + ".incomplete" for fname in tfrecord_paths]
    writers = [tf.io.TFRecordWriter(fname) for fname in tmp_file_paths]

    shard = 0
    total_line = Util.get_file_line(meta_file)

    logger.info("Meta file: %s", meta_file)
    with open(meta_file) as json_file:
      for json_line in json_file:
        Util.printProgress(counter + 1, total_line, 'Progress:', '', 2, 80)
        spec = json.loads(json_line.strip())
        if config.decoding_from_npy:
          feats = np.load(Util.get_file_path(data_path, spec["key"]))
        else:
          feats = np.load(Util.get_file_path(data_path, spec["key"]
                                             +"." + config.feat_type + ".npy"))

        if is_debug:
          plt.imshow(feats)
          plt.show()

        # normalizing features per each speaker
        if config.prep_data_name == 'wsj':
          modified_key = spec["key"].replace("//", "/")
          utt_split_idx = 4 if modified_key.find("wsj64k") == -1 else 5
          if not config.decoding_from_npy:
            spk_id = modified_key.split("/")[utt_split_idx]
          utt_id = spec["key"].split("/")[-1].split(".")[0]
        elif config.prep_data_name == 'libri':
          if not config.decoding_from_npy:
            spk_id = spec["key"].split("/")[-1].split("-")[0] + "-" + \
                     spec["key"].split("/")[-1].split("-")[1]
          utt_id = spec["key"].split("/")[-1].split(".")[0]
        elif config.prep_data_name == 'timit':
          if config.decoding_from_npy:
            id = spec["key"].split("/")[-1].split(".npy")[0].split("_")
            utt_id = id[0] + "-" + id[1]
          else:
            spk_id = spec["key"].split("/DR")[1].split("/")[1]
            utt_id = spk_id + "-" + spec["key"].split("/")[-1].split(".")[0]

        if cmvn:
          feats = (feats - cmvn[spk_id][0] + 1e-14) / (cmvn[spk_id][1] + 1e-14)

        if is_debug:
          plt.imshow(feats)
          plt.show()

        if feats.shape[1] != config.feat_dim:
          logger.critical("feature dimension option is incorrect! please "
                          "check it, generated feature dimension: %d, "
                          "the given feature dimension: %d",
                          feats.shape[1],
                          config.feat_dim)
          sys.exit(ExitCode.INVALID_OPTION)

        #tfrecord_in_text = open("check_saving","w")
        int_seq = Util.get_int_seq(spec["text"], is_char=is_char, vocab=vocab)
        example = tf.train.Example(features=tf.train.Features(feature={
            "target_label": _int64_list_feat(int_seq),
            "input_speech": _float_list_feat(np.array(feats).flatten()),
            "input_length": _int64_list_feat([feats.shape[0]]),
            "target_length": _int64_list_feat([len(int_seq)]),
            "utt_id": _string_list_feat([utt_id.encode('utf-8')])
        }))

        writers[shard].write(example.SerializeToString())

        if is_debug:
          np.savetxt("debug_feat.np", feats)
          tfrecord_in_text = open("debug_tfrecord.txt", "w")
          target_label = " ".join(str(x) for x in int_seq)
          tfrecord_in_text.write("input_length: %d\n"%feats.shape[0])
          tfrecord_in_text.write("target_label: %s\n" % target_label)
          tfrecord_in_text.write("target_length: %d\n"%len(int_seq))
          tfrecord_in_text.close()

        # round robin order indexing
        shard = (shard + 1) % total_shards
        counter += 1

    for writer in writers:
      writer.close()

    for tmp_name, final_name in zip(tmp_file_paths, tfrecord_paths):
      tf.io.gfile.rename(tmp_name, final_name)

    logger.info("Saved %d Examples in %.2f seconds", counter,
                (Util.currentTimeMillis() - current_milli_time)/1000.0)
  return tfrecord_paths, counter


def shuffle_records(tfrecord_file):
  """Shuffle records in a single file."""
  # Rename file prior to shuffling
  tmp_fname = tfrecord_file + ".unshuffled"
  tf.io.gfile.rename(tfrecord_file, tmp_fname)

  reader = tf.compat.v1.io.tf_record_iterator(tmp_fname)
  records = []
  for record in reader:
    records.append(record)
  random.shuffle(records)

  # Write shuffled records to original file name
  with tf.io.TFRecordWriter(tfrecord_file) as shuffled_writer:
    for record in records:
      shuffled_writer.write(record)

  tf.io.gfile.remove(tmp_fname)


def main():
  # Loading config and logger
  logger = Logger(name="TFRecord", level=Logger.DEBUG).logger
  logger.info("Tensorflow Version: %s", tf.__version__)
  config = ParseOption(sys.argv, logger).args

  # Loading cmvn per each speaker
  if config.decoding_from_npy:
    cmvn = None
  else:
    cmvn_path = Util.get_file_path(config.path_base, config.path_cmvn_ptrn)
    cmvn, spk_n = Util.load_cmvn(cmvn_paths=cmvn_path,
                                 dataset=config.prep_data_name)
    logger.info("Feature mean and variance for %d speakers from %s", spk_n,
                cmvn_path)

  # logger, config, data_set, total_shards, data_name
  tfrecord_files = None
  examples = 0
  if config.path_train_json is not None:
    tfrecord_files, examples = convert_to_tfrecord(logger, config, Tag.TRAIN,
                                                   cmvn)
  if config.path_valid_json is not None:
    convert_to_tfrecord(logger, config, Tag.VALID, cmvn)
  if config.path_test_json is not None:
    convert_to_tfrecord(logger, config, Tag.TEST, cmvn)

  if tfrecord_files:
    logger.info("Shuffling training data.")
    start_millis = Util.currentTimeMillis()
    for idx, tfrecord_file in enumerate(tfrecord_files):
      Util.printProgress(idx + 1, len(tfrecord_files), 'Progress:', '', 2, 80)
      shuffle_records(tfrecord_file)
    logger.info("Shuffled %d Examples in %.2f seconds", examples,
                (Util.currentTimeMillis() - start_millis)/1000.0)

if __name__ == "__main__":
  main()
