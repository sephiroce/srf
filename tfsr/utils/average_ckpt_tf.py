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

# pylint: disable=import-error, too-many-locals, too-many-statements
# pylint: disable=pointless-string-statement, no-member

"""
average_ckpt.py: Tensorflow 2.1 averaging model weights.
I referred to https://stackoverflow.com/questions/48212110/average-weights-in-keras-models
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"


import os
import sys
import shutil
import numpy as np

import tensorflow as tf
from tfsr.helper.common_helper import ParseOption, Logger
from tfsr.helper.misc_helper import Util
import tfsr.helper.train_helper as th
from tfsr.trainer_tf import ConvEncoder


def main():
  # pylint: disable=too-many-branches
  # Initializing
  Util.prepare_device()
  logger = Logger(name="speech_transformer", level=Logger.DEBUG).logger
  config = ParseOption(sys.argv, logger).args

  # Loading a vocabulary
  _, _, dec_in_dim, dec_out_dim = \
    Util.load_vocab(Util.get_file_path(config.path_base, config.path_vocab),
                    logger)
  dec_out_dim = dec_in_dim + 1
  logger.info("The modified output Dimension %d", dec_out_dim)

  # Model selection
  # pylint: disable=invalid-name
  model = ConvEncoder(config.model_encoder_num, config.model_dimension,
                      config.model_att_head_num, config.model_inner_dim,
                      config.feat_dim,
                      config.train_inp_dropout, config.train_inn_dropout,
                      config.train_att_dropout, config.train_res_dropout,
                      config.model_conv_filter_num,
                      config.model_conv_layer_num,
                      config.model_initializer, dec_out_dim)

  # Setting optimizer and checkpoint manager
  optimizer = th.get_optimizer(config)

  # Creating or loading a check point
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  ckpt_manager = \
    tf.train.CheckpointManager(ckpt, config.path_ckpt, max_to_keep=None)

  ckpts = []
  for ckpt in ckpt_manager.checkpoints:
    if config.train_max_epoch == 0 or \
      int(ckpt.split("-")[-1]) <= config.train_max_epoch:
      ckpts.append(ckpt)

  optimizer = th.get_optimizer(config)

  models = []
  for ckpt_path in ckpts[-config.model_average_num:]:
    logger.info(ckpt_path)
    model = ConvEncoder(config.model_encoder_num, config.model_dimension,
                        config.model_att_head_num, config.model_inner_dim,
                        config.feat_dim,
                        config.train_inp_dropout, config.train_inn_dropout,
                        config.train_att_dropout, config.train_res_dropout,
                        config.model_conv_filter_num,
                        config.model_conv_layer_num,
                        config.model_initializer, dec_out_dim)

    # Creating or loading a check point
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(ckpt_path).expect_partial()

    dummy_feats = tf.random.uniform([1, 20, config.feat_dim])
    dummy_in_len = tf.ones(1) * 20
    model(dummy_feats, input_lengths=dummy_in_len, is_training=False,
          mask=None, attention_penalty_mask=None)

    models.append(model)

  logger.info("Total %d models were loaded.", len(models))

  # Computing averaged weights
  weights = list()
  for model in models:
    weights.append(model.get_weights())

  new_weights = list()
  for weights_list_tuple in zip(*weights):
    new_weights.append(
        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
    )

  # Saving
  model = ConvEncoder(config.model_encoder_num, config.model_dimension,
                      config.model_att_head_num, config.model_inner_dim,
                      config.feat_dim,
                      config.train_inp_dropout, config.train_inn_dropout,
                      config.train_att_dropout, config.train_res_dropout,
                      config.model_conv_filter_num,
                      config.model_conv_layer_num,
                      config.model_initializer, dec_out_dim)

  dummy_feats = tf.random.uniform([10, 20, config.feat_dim])
  dummy_in_len = tf.ones(10) * 20
  model(dummy_feats, input_lengths=dummy_in_len, is_training=False,
        mask=None, attention_penalty_mask=None)

  model.set_weights(weights[0])

  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  if os.path.exists(config.path_ckpt + "/avg"):
    shutil.rmtree(config.path_ckpt + "/avg")

  ckpt_manager = \
    tf.train.CheckpointManager(ckpt, config.path_ckpt + "/avg", max_to_keep=1)

  logger.info("Saved to %s", ckpt_manager.save())

if __name__ == "__main__":
  main()
