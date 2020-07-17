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

# pylint: disable=import-error, no-member, too-many-locals


"""
trainer_sr.py: a main function of a sequence router implementation
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import sys
import time

import tensorflow as tf
from tfsr.helper.common_helper import ParseOption, Logger
from tfsr.helper.misc_helper import Util
import tfsr.helper.data_helper as dh
from tfsr.model.sequence_router import SequenceRouter
import tfsr.helper.train_helper as th

def process_train_step(config, inputs, model, optimizer, loss_state,
                       frame_state, n_gpus, blank_idx):
  # pylint: disable=too-many-arguments
  feats, labels, inp_len, tar_len = inputs
  batch = tf.shape(feats)[0]
  global_batch_size = tf.cast(batch * n_gpus, tf.float32)
  in_len_div = config.model_conv_layer_num ** config.model_conv_stride

  max_feat_len = tf.math.reduce_max(inp_len, keepdims=False)
  feats = feats[:, :max_feat_len, :]

  with tf.GradientTape() as tape:
    y_pred = model(feats, input_lengths=inp_len, training=True)
    pe_loss = tf.nn.ctc_loss(labels, y_pred, tar_len,
                             tf.math.ceil(inp_len / in_len_div),
                             logits_time_major=False, blank_index=blank_idx)
    loss = tf.nn.compute_average_loss(pe_loss,
                                      global_batch_size=global_batch_size)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  loss_state.update_state(pe_loss)
  frame_state.update_state(tf.math.reduce_sum(inp_len))


def process_valid_step(config, inputs, model, loss_state, blank_idx):
  # pylint: disable=too-many-arguments
  feats, labels, inp_len, tar_len = inputs
  in_len_div = config.model_conv_layer_num ** config.model_conv_stride

  max_feat_len = tf.math.reduce_max(inp_len, keepdims=False)
  feats = feats[:, :max_feat_len, :]
  y_pred = model(feats, input_lengths=inp_len, training=False)
  pe_loss = tf.nn.ctc_loss(labels, y_pred, tar_len,
                           tf.math.ceil(inp_len / in_len_div),
                           logits_time_major=False, blank_index=blank_idx)
  loss_state.update_state(pe_loss)


def process_test_step(config, inputs, model):
  # pylint: disable=too-many-arguments
  feats, _, inp_len, _, utt_id = inputs
  in_len_div = config.model_conv_layer_num ** config.model_conv_stride
  max_feat_len = tf.math.reduce_max(inp_len, keepdims=False)
  feats = feats[:, :max_feat_len, :]
  y_pred = model(feats, input_lengths=inp_len, training=False)
  y_pred = tf.transpose(y_pred, [1, 0, 2])
  hypos, _ = tf.nn.ctc_beam_search_decoder(y_pred, inp_len // in_len_div,
                                           beam_width=config.decoding_beam_width,
                                           top_paths=1)
  #tf.print(tf.reshape(y_pred, [137, 63]),
  # summarize=10000000,
  #           output_stream="file://dr_2_ctc.txt")
  tf.print("UTTID:", utt_id)
  for hyp in hypos:
    tf.print(hyp, summarize=1000)


def main():
  #pylint: disable=too-many-branches, too-many-statements
  # Initializing toolkit
  Util.prepare_device()
  logger = Logger(name="speech_transformer", level=Logger.DEBUG).logger
  config = ParseOption(sys.argv, logger).args

  _, _, dec_in_dim, _ = Util.load_vocab(Util.get_file_path(config.path_base,
                                                           config.path_vocab), logger)
  dec_out_dim = dec_in_dim + 1
  blank_idx = dec_in_dim
  logger.info("The modified output Dimension %d, blank index %d", dec_out_dim,
              dec_in_dim)

  # Setting a distribute strategy
  strategy = tf.distribute.MirroredStrategy()
  num_gpus = strategy.num_replicas_in_sync

  # Creates dataset
  train_ds, valid_ds = dh.create_ds_for_training(config, logger, num_gpus)
  test_ds = None
  if config.train_max_epoch == 0:
    test_ds = dh.create_ds_for_evaluation(config, logger)

  # Creates measurements
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  valid_loss = tf.keras.metrics.Mean(name='valid_loss')
  num_feats = tf.keras.metrics.Mean(name='feature_number')

  # Distributed loop
  with strategy.scope():
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    valid_ds = strategy.experimental_distribute_dataset(valid_ds)
    if config.train_max_epoch == 0:
      test_ds = strategy.experimental_distribute_dataset(test_ds)

    # Creating a model & preparing to train
    model = SequenceRouter(config, logger, dec_out_dim)
    opti = th.get_optimizer(config)
    ckpt_mgr, epoch_offset = Util.load_checkpoint(config, logger, model, opti)

    @tf.function
    def distributed_train_step(dataset):
      index = 0
      for example in dataset:
        args = (config, example, model, opti, train_loss, num_feats,
                num_gpus, blank_idx)
        strategy.run(process_train_step, args=args)
        if index % 50 == 0 and index > 0:
          tf.print("Average Loss", train_loss.result())
        index += 1

    @tf.function
    def distributed_valid_step(dataset):
      for example in dataset:
        args = (config, example, model, valid_loss, blank_idx)
        strategy.run(process_valid_step, args=args)

    @tf.function
    def distributed_test_step(dataset):
      for example in dataset:
        args = (config, example, model)
        strategy.run(process_test_step, args=args)

    if config.train_max_epoch > 0:
      distributed_valid_step(valid_ds)
      model.summary()

    pre_loss = 1e+9
    tolerance = 0
    for epoch in range(epoch_offset, config.train_max_epoch):
      # Initializing the measurements
      train_loss.reset_states()
      valid_loss.reset_states()
      num_feats.reset_states()

      prev = time.time()
      distributed_train_step(train_ds)
      logger.info('Epoch {:03d} Train Loss {:.4f}, {:.3f} secs, '
                  '{:d} feats/step, {:d}/{:d} steps'.
                  format(epoch + 1, train_loss.result(), time.time() -
                         prev, int(num_feats.result()) * num_gpus,
                         opti.iterations.numpy(), config.train_max_step))

      prev = time.time()
      distributed_valid_step(valid_ds)
      better = valid_loss.result() - pre_loss <= (pre_loss * 0.01)

      if better:
        tolerance = 0
      else:
        tolerance += 1

      logger.info('Epoch {:03d} Valid Loss {:.4f}, {:.3f} secs{}'.
                  format(epoch + 1, valid_loss.result(), time.time() - prev,
                         ", improved" if better else ", tolerance %d"%tolerance))
      pre_loss = valid_loss.result()

      if 0 < config.train_es_tolerance <= tolerance:
        logger.info('early stopped!')
        break

      # Saving the last checkpoint if it was not id.
      if config.train_ckpt_saving_per > 0:
        ckpt_path = ckpt_mgr.save()
        logger.info('Saving a ckpt for the last epoch at %s', ckpt_path)
      else:
        logger.warning("Not saved since train-ckpt-saving-per is %d, it needs to"
                       " be bigger than 0 if you want save checkpoints",
                       config.train_ckpt_saving_per)

    if config.train_max_epoch == 0:
      logger.info("Recognizing speeches")
      prev = time.time()
      distributed_test_step(test_ds)
      logger.info("%.3f secs elapsed", (time.time() - prev))


if __name__ == "__main__":
  main()
