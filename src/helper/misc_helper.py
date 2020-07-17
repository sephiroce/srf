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

# pylint: disable=too-many-arguments, too-many-locals, import-error
# pylint: disable=unused-argument, too-many-instance-attributes
# pylint: disable=too-few-public-methods, attribute-defined-outside-init

"""misc_helper.py: Utilities for ASR systems"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import glob
import os
import sys
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tfsr.helper.common_helper import Constants, ExitCode

class Util:
  @staticmethod
  def currentTimeMillis():
    return int(round(time.time() * 1000))

  @staticmethod
  def make_dir(path):
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)

  @staticmethod
  def get_file_line(fname):
    return sum(1 for _ in open(fname))

  @staticmethod
  def all_exist(file_names):
    """Returns true if all files in the list exist."""
    for file_name in file_names:
      if not tf.io.gfile.exists(file_name):
        return False
    return True

  @staticmethod
  def ensure_dir(log_dir):
    if not tf.io.gfile.exists(log_dir):
      tf.io.gfile.makedirs(log_dir)

  @staticmethod
  def get_file_path(data_path, file_path):
    """
    In order to handle both absolute paths and relative paths, it returns an
    exist path among combinations of data_path and file_path.

    :param data_path: base path
    :param file_path: file path
    :return: a existed file path
    """
    data_path = data_path.strip()
    file_path = file_path.strip()
    return file_path if os.path.isfile(file_path) \
      else data_path + "/" + file_path

  @staticmethod
  def load_vocab(vocab_path, logger=None):
    vocab = list()
    with open(vocab_path) as vocab_file:
      for line in vocab_file:
        token = line.strip()
        if token == Constants.SPACE:
          vocab.append(" ")
        else:
          vocab.append(token.strip())

    if vocab[-1] != Constants.BOS:
      msg = "Last index must be BOS: %s, but %s"%(Constants.BOS, vocab[-1])
      if logger is None:
        print(msg)
      else:
        logger.critical(msg)

    str_to_int = dict()
    for token_id, token in enumerate(vocab):
      str_to_int[token] = token_id

    dec_in_dim = len(vocab)
    dec_out_dim = dec_in_dim - 1 if Constants.BOS in str_to_int else dec_in_dim

    msg = "Decoder Input Dim: %d, Output Dim %d"%(dec_in_dim, dec_out_dim)
    if logger is None:
      print(msg)
    else:
      logger.info(msg)

    return vocab, str_to_int, dec_in_dim, dec_out_dim

  @staticmethod
  def printProgress(iteration, total, prefix='', suffix='', decimals=1,
                    bar_len=100):
    """
    url: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-
         console
    url: https://yujuwon.tistory.com/entry/python%EC%97%90%EC%84%9C-Progressbar-
         %ED%91%9C%ED%98%84%ED%95%98%EA%B8%B0

    :param iteration: index
    :param total: total number
    :param prefix:
    :param suffix:
    :param decimals: decimal of the float point
    :param bar_len:
    :return:
    """
    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_len * iteration / float(total)))
    progress_bar = '#' * filled_length + '-' * (bar_len - filled_length)
    percent_symbol = "%"
    sys.stdout.write('\r%s |%s| %s%s (%d/%d) %s' % (prefix, progress_bar,
                                                    percent, percent_symbol,
                                                    iteration, total, suffix))
    if iteration == total:
      sys.stdout.write('\n')
    sys.stdout.flush()

  @staticmethod
  def load_checkpoint(config, logger, model, optimizer):
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    max_to_keep = config.model_ckpt_max_to_keep
    if max_to_keep < 0:
      max_to_keep = None
    ckpt_manager = \
      tf.train.CheckpointManager(ckpt, config.path_ckpt, max_to_keep=max_to_keep)

    loaded_ckpt = ""
    if config.path_ckpt_epoch is not None and config.path_ckpt_epoch > 0:
      loaded_ckpt = config.path_ckpt + "/ckpt-%d" % config.path_ckpt_epoch
    elif ckpt_manager.latest_checkpoint:
      loaded_ckpt = ckpt_manager.latest_checkpoint

    if "ckpt" in loaded_ckpt:
      epoch_offset = int(loaded_ckpt.split("-")[-1])
      ckpt.restore(loaded_ckpt).expect_partial()
    else:
      epoch_offset = 0
      loaded_ckpt = None

    logger.info("Loaded ckpt: %s", loaded_ckpt)

    return ckpt_manager, epoch_offset

  @staticmethod
  def prepare_device():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
      tf.config.experimental.set_memory_growth(device, True)

  @staticmethod
  def plot_attention_weights(attention, layer, title):
    idx = 1
    fig = plt.figure(figsize=(16, 8))
    plt.title(title + "%d"%idx)
    idx += 1

    if layer:
      attention = tf.squeeze(attention[layer], axis=0)
    else:
      attention = tf.squeeze(attention, axis=0)

    for head in range(attention.shape[0]):
      fig_subplot = fig.add_subplot(2, 2, head + 1)

      # plot the attention weights
      fig_subplot.matshow(attention[head][:-1, :], cmap='viridis')

    plt.tight_layout()
    plt.show()

  @staticmethod
  def load_cmvn(cmvn_paths, dataset='wsj'):
    cmvn = dict()
    for cmvn_file in glob.glob(cmvn_paths):
      if dataset == 'wsj':
        spk_id_len = 3
        cmvn[cmvn_file.split("spk_")[1][:spk_id_len]] = np.loadtxt(cmvn_file)
      elif dataset == 'timit':
        spk_id_len = 5
        cmvn[cmvn_file.split("spk_")[1][:spk_id_len]] = np.loadtxt(cmvn_file)
      elif dataset == 'libri':
        cmvn[cmvn_file.split("spk_")[1].split(".")[0]] = np.loadtxt(cmvn_file)

    return cmvn, len(cmvn)

  @staticmethod
  def get_int_seq(text, is_char, vocab):
    int_seq = list()
    text = text.strip().replace("  ", " ")
    if is_char:
      for char in text:
        if char in vocab:
          int_seq.append(vocab[char])
        elif char == ' ':
          int_seq.append(vocab[Constants.SPACE])
        else:
          print(vocab)
          print("%s is not in vocab"%char)
          sys.exit(ExitCode.NOT_SUPPORTED)
    else:
      for bpe in text.split(" "):
        int_seq.append(vocab[bpe])
    return int_seq
