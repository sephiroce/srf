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

# pylint: disable=import-error, too-many-locals

"""data_helper: creating datasets"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import os
import tfsr.helper.train_helper as th
from tfsr.data import load_speech_data

def create_ds_for_evaluation(config, logger):
  test_file_ptrn = os.path.join(config.path_base, config.path_test_ptrn)
  map_data_fn = load_speech_data.map_data_for_transformer_utt_id_fn

  logger.info("Batch size for test will be set to 1")

  test_ds = \
    load_speech_data.create_ds_batch_for_test(file_pattern=test_file_ptrn,
                                              num_parallel_calls=6,
                                              batch_size=1,
                                              max_inp=config.prep_max_inp,
                                              max_tar=config.prep_max_tar)

  test_ds = test_ds.map(lambda x, y, a, b, c:
                        map_data_fn(x, y, a, b, c, config.feat_dim),
                        num_parallel_calls=6)

  return test_ds

def create_ds_for_training(config, logger, num_gpus):
  train_file_ptrn = os.path.join(config.path_base, config.path_train_ptrn)
  valid_file_ptrn = os.path.join(config.path_base, config.path_valid_ptrn)
  map_data_fn = load_speech_data.map_data_for_transformer_fn

  if config.train_batch_dynamic:
    assert config.train_batch_frame is not None and config.train_batch_frame \
           > 0
    bucket_boundaries, bucket_batch_sizes = \
      th.get_bucket_info(config.train_batch_frame, num_gpus, 300, 10000, 200)

    logger.info('bucket_boundaries: [%s]', ', '.join(map(str, bucket_boundaries)))
    logger.info('bucket_batch_sizes: [%s]', ', '.join(map(str,
                                                          bucket_batch_sizes)))

    train_ds = load_speech_data.create_ds_bucket(file_pattern=train_file_ptrn,
                                                 num_parallel_calls=6,
                                                 shuffle=True,
                                                 repeat=1,
                                                 bucket_boundaries=bucket_boundaries,
                                                 bucket_batch_sizes=bucket_batch_sizes,
                                                 max_inp=config.prep_max_inp,
                                                 max_tar=config.prep_max_tar)

    valid_ds = load_speech_data.create_ds_bucket(file_pattern=valid_file_ptrn,
                                                 num_parallel_calls=6,
                                                 shuffle=False,
                                                 repeat=1,
                                                 bucket_boundaries=bucket_boundaries,
                                                 bucket_batch_sizes=bucket_batch_sizes,
                                                 max_inp=config.prep_max_inp,
                                                 max_tar=config.prep_max_tar)
  else:
    assert config.train_batch_size is not None and config.train_batch_size > 0

    train_ds = load_speech_data.create_ds_batch_for_train(file_pattern=train_file_ptrn,
                                                          num_parallel_calls=6,
                                                          shuffle=True,
                                                          repeat=1,
                                                          batch_size=config.train_batch_size,
                                                          max_inp=config.prep_max_inp,
                                                          max_tar=config.prep_max_tar)

    valid_ds = load_speech_data.create_ds_batch_for_train(file_pattern=valid_file_ptrn,
                                                          num_parallel_calls=6,
                                                          shuffle=False,
                                                          repeat=1,
                                                          batch_size=config.train_batch_size,
                                                          max_inp=config.prep_max_inp,
                                                          max_tar=config.prep_max_tar)

  train_ds = \
    train_ds.map(lambda x, y, a, b: map_data_fn(x, y, a, b, config.feat_dim),
                 num_parallel_calls=6)

  valid_ds = \
    valid_ds.map(lambda x, y, a, b: map_data_fn(x, y, a, b, config.feat_dim),
                 num_parallel_calls=6)

  return train_ds, valid_ds
