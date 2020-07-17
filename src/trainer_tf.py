import sys
import time
import math

import tensorflow as tf
from tfsr.helper.common_helper import ParseOption, Logger
from tfsr.helper.misc_helper import Util
import tfsr.helper.data_helper as dh
import tfsr.helper.model_helper as mh
import tfsr.helper.train_helper as th
import tfsr.model.block as block
from tfsr.model.sequence_router import CapsulationLayer as ConvLayer

class ConvEncoder(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, feat_dim,
               input_dropout, inner_dropout, residual_dropout,
               attention_dropout, nfilt, cnn_n, init,
               vocab_n):
    super(ConvEncoder, self).__init__()

    self.d_model = d_model
    self.dff = dff
    self.num_layers = num_layers
    self.attention_dropout = attention_dropout
    self.num_heads = num_heads
    self.residual_dropout = residual_dropout

    self.enc_layers = [block.EncoderBlock(self.d_model, self.num_heads, dff,
                                          inner_dropout, residual_dropout,
                                          attention_dropout, init)
                       for _ in range(num_layers)]

    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # TODO: stride and filter number are hard coded.
    self.stride = 2
    kernel_size = 3
    self.cnn_n = cnn_n
    self.mask2_layer = tf.keras.layers.Lambda(mh.feat_mask2)

    self.conv = ConvLayer(cnn_n, nfilt, kernel_size, self.stride, init)
    self.reshape_to_ffwd = \
      tf.keras.layers.Reshape((-1, math.ceil(feat_dim / (2 * 2)) * nfilt),
                              name="reshape_to_ffwd")

    self.linear_projection = tf.keras.layers.Dense(d_model,
                                                   activation='linear',
                                                   kernel_initializer=mh.get_init(init))

    self.input_dropout = tf.keras.layers.Dropout(rate=input_dropout) # (batch_size, input_seq_len, d_model)
    self.proj = tf.keras.layers.Dense(vocab_n) # (batch_size, input_seq_len, vocab)

  def call(self, inputs, **kwargs):
    # pylint: disable=arguments-differ
    """
    from the Speech-Transformer paper,
      We firstly stack two 3×3 CNN layers with stride 2 for both time and
      frequency dimensions to prevent the GPU memory overflow and produce the
      approximate hidden representation length with the character length.
    """
    in_len_div = 2 * 2
    inp_len = kwargs["input_lengths"]
    training = kwargs["is_training"]
    mask = kwargs["mask"]
    att_penalty = kwargs["attention_penalty_mask"]

    out, _, _ = self.conv(inputs, input_lengths=inp_len)
    out = self.reshape_to_ffwd(out)
    out = self.linear_projection(out)
    embeddings = self.mask2_layer([out, inp_len, in_len_div])

    # adding positional encodings to prepared feature sequences
    seq_len = tf.shape(embeddings)[1]
    embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    embeddings += mh.get_pos_enc(seq_len, self.d_model)
    embeddings = self.input_dropout(embeddings, training=training)

    for idx, enc_layer in enumerate(self.enc_layers):
      embeddings = enc_layer(embeddings, is_training=training,
                             mask=mask, attention_penalty_mask=att_penalty)

    embeddings = self.layernorm(embeddings) # (batch_size, input_seq_len, d_model)
    return self.proj(embeddings) # (batch_size, input_seq_len, vocab)


def process_train_step(config, inputs, model, optimizer, loss_state,
                       frame_state, att_pen, n_gpus, blank_idx):
  # pylint: disable=too-many-arguments
  feats, labels, inp_len, tar_len = inputs
  batch_size = tf.shape(feats)[0]
  global_batch_size = tf.cast(batch_size * n_gpus, tf.float32)
  in_len_div = 2 * 2

  feats, _, _, enc_pad_mask, _ = \
    th.prep_process(labels, inp_len, tar_len, feats, in_len_div)

  enc_att_pen = None
  if att_pen is not None:
    enc_att_pen, _, _ = att_pen.create_pens(config,
                                            tf.math.ceil(inp_len / in_len_div),
                                            tar_len)

  with tf.GradientTape() as tape:
    y_pred = model(feats, input_lengths=inp_len, is_training=True,
                   mask=enc_pad_mask, attention_penalty_mask=enc_att_pen)
    pe_loss = tf.nn.ctc_loss(labels, y_pred, tar_len,
                             tf.math.ceil(inp_len / in_len_div),
                             logits_time_major=False, blank_index=blank_idx)
    loss = tf.nn.compute_average_loss(pe_loss,
                                      global_batch_size=global_batch_size)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  loss_state.update_state(pe_loss)
  frame_state.update_state(tf.math.reduce_sum(inp_len))


def process_valid_step(config, inputs, model, loss_state, att_pen, blank_idx):
  # pylint: disable=too-many-arguments
  feats, labels, inp_len, tar_len = inputs
  in_len_div = 2 * 2

  feats, _, _, enc_pad_mask, _ = \
    th.prep_process(labels, inp_len, tar_len, feats, in_len_div)

  enc_att_pen = None
  if att_pen is not None:
    enc_att_pen, _, _ = att_pen.create_pens(config,
                                            tf.math.ceil(inp_len / in_len_div),
                                            tar_len)

  y_pred = model(feats, input_lengths=inp_len, is_training=False,
                 mask=enc_pad_mask, attention_penalty_mask=enc_att_pen)
  pe_loss = tf.nn.ctc_loss(labels, y_pred, tar_len,
                           tf.math.ceil(inp_len / in_len_div),
                           logits_time_major=False, blank_index=blank_idx)
  loss_state.update_state(pe_loss)


def process_test_step(config, inputs, model, att_pen):
  # pylint: disable=too-many-arguments
  feats, labels, inp_len, tar_len, utt_id = inputs
  in_len_div = 2 * 2

  feats, _, _, enc_pad_mask, _ = \
    th.prep_process(labels, inp_len, tar_len, feats, in_len_div)

  enc_att_pen = None
  if att_pen is not None:
    enc_att_pen, _, _ = att_pen.create_pens(config,
                                            tf.math.ceil(inp_len / in_len_div),
                                            tar_len)

  y_pred = model(feats, input_lengths=inp_len, is_training=False,
                  mask=enc_pad_mask, attention_penalty_mask=enc_att_pen)
  y_pred = tf.transpose(y_pred, [1,0,2])
  hypos, _ = tf.nn.ctc_beam_search_decoder(y_pred, inp_len // in_len_div,
                                           beam_width=config.decoding_beam_width,
                                           top_paths=1)
  tf.print("UTTID:", utt_id)
  for hyp in hypos:
    tf.print(hyp, summarize=1000)


def main():
  #pylint: disable=too-many-branches

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
  test_ds = dh.create_ds_for_evaluation(config, logger)

  # Creates measurements
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  num_feats = tf.keras.metrics.Mean(name='feature_number')
  valid_loss = tf.keras.metrics.Mean(name='valid_loss')

  # Distributed loop
  with strategy.scope():
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    valid_ds = strategy.experimental_distribute_dataset(valid_ds)
    test_ds = strategy.experimental_distribute_dataset(test_ds)

    # Creating a model & preparing to train
    att_pen = mh.create_attention_penalty(config, logger)
    model = ConvEncoder(config.model_encoder_num, config.model_dimension,
                        config.model_att_head_num, config.model_inner_dim,
                        config.feat_dim,
                        config.train_inp_dropout, config.train_inn_dropout,
                        config.train_att_dropout, config.train_res_dropout,
                        config.model_conv_filter_num,
                        config.model_conv_layer_num,
                        config.model_initializer, dec_out_dim)
    opti = th.get_optimizer(config)
    ckpt_mgr, epoch_offset = Util.load_checkpoint(config, logger, model,
                                                      opti)
    @tf.function
    def distributed_train_step(dataset):
      index = 0
      for example in dataset:
        args = (config, example, model, opti, train_loss,
                num_feats, att_pen, num_gpus, blank_idx)
        strategy.run(process_train_step, args=args)
        if index % 50 == 0:
          tf.print("Average Loss", train_loss.result())
        index += 1

    @tf.function
    def distributed_valid_step(dataset):
      for example in dataset:
        args = (config, example, model, valid_loss, att_pen, blank_idx)
        strategy.run(process_valid_step, args=args)

    @tf.function
    def distributed_test_step(dataset):
      for example in dataset:
        args = (config, example, model, att_pen)
        strategy.run(process_test_step, args=args)

    distributed_valid_step(valid_ds)
    model.summary()

    pre_loss = 1e+9
    tolerance = 0
    for epoch in range(epoch_offset, config.train_max_epoch):
      train_loss.reset_states()
      num_feats.reset_states()
      valid_loss.reset_states()

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