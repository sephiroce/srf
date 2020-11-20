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

# pylint: disable=too-few-public-methods, too-many-locals, no-member,
# pylint: disable=too-many-statements

"""
common_helper.py: global functionaries for python programs
"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

from enum import Enum

import argparse
import logging
import os
import sys

class Tag:
  TRAIN = 'train'
  VALID = 'valid'
  TEST = 'test'

class Constants: # pylint: disable=no-init
  # Special tokens
  PAD_CHAR = 'p'
  PAD_WORD = "<PADDING_SYMBOL>"
  SPACE = '<SPACE>'
  UNK = '<unk>'
  UNKs = ['<UNK>', '<unk>']
  EOS = '$'
  BOS = '@'
  EPS = 1e-14
  NOISE_SYM = 'n'
  INF = 1e9

  # Token unit
  WORD = 'word'
  CHAR = 'char'

  # Json
  DURATION = 'duration'
  KEY = 'key'
  TEXT = 'text'

  # Model
  INPUT_TRANS = 'INPUT_TRANS'
  INPUT_PREDS = 'INPUT_PREDS'
  INPUT_LABEL = 'INPUT_LABEL'
  INPUT_INLEN = 'INPUT_INLEN'
  INPUT_LBLEN = 'INPUT_LBLEN'
  TARGET_LABEL = 'TARGET_LABEL'

  # Smoothing
  SM_NEIGHBOR = 'neighbor'
  SM_LABEL = 'label'

  # Initializer
  INIT_GLOROT = 'glorot_uniform'
  INIT_FANAVG = 'fan_avg'
  INIT_UNIFORM = 'uniform'

class ExitCode(Enum):
  NO_DATA = 0
  NOT_SUPPORTED = 1
  INVALID_OPTION = 11
  INVALID_CONVERSION = 12
  INVALID_NAME = 13
  INVALID_NAME_OF_CONFIGURATION_FILE = 14
  INVALID_FILE_PATH = 15
  INVALID_DICTIONARY = 16
  INVALID_CONDITION = 17

class Logger:
  """
  !!Usage: please create a logger with one line as shown in the example below.
    logger = Logger(name = "word2vec", level = Logger.DEBUG).logger
    logger.debug('debug message')
    logger.info('info message')
    logger.warn('warn message')
    logger.error('error message')
    logger.critical('critical message')

  This logger print out logging messages similar to the logging message of tensorflow.
  2018-07-01 19:35:33.945120: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406]
  2018-07-20 16:23:08.000295: I kmlm_common.py:94] Configuration lists:

  TO-DO:
  verbose level
  """
  DEBUG = logging.DEBUG
  NOTSET = logging.NOTSET
  INFO = logging.INFO
  WARN = logging.WARN
  ERROR = logging.ERROR
  CRITICAL = logging.CRITICAL

  def __init__(self, name="__default__", level=logging.NOTSET):
    self.logger = logging.getLogger(name)
    self.logger.setLevel(level)
    handle = logging.StreamHandler()
    handle.setLevel(level)
    formatter = logging.Formatter('%(asctime)s: %(levelname).1s '
                                  '%(filename)s:%(lineno)d] %(message)s')
    formatter.default_msec_format = '%s.%06d'
    handle.setFormatter(formatter)
    self.logger.propagate = False
    self.logger.addHandler(handle)

class ParseOption:
  """
  it merges options from both an option file and command line into python option
  """
  def __init__(self, argv, logger, is_print_opts=True):
    self.logger = logger

    parser = self.build_parser()

    command_keys = set()
    for command_key in argv[1:]:
      command_keys.add(command_key.replace("-", "_")[2:command_key.find("=")])

    if len(argv) > 1:
      # args from command line
      command_args = parser.parse_args(argv[1:])

      # args from config file
      if command_args.config is not None and \
              not command_args.config.endswith(".conf"):
        self.logger.critical("The the extension of configuration file must be "
                             "conf, but %s" %command_args.config)
        sys.exit(ExitCode.INVALID_NAME_OF_CONFIGURATION_FILE)

      data_path = command_args.path_base
      config_args = None
      if command_args.config:
        file_path = command_args.config
        if data_path and not os.path.exists(file_path):
          file_path = data_path + "/" + file_path

        config_args = parser.parse_args(["@" + file_path])

      # merge command args and file args into args
      command_dict = vars(command_args)
      if command_args.config:
        if "config" not in command_keys:
          self.logger.critical("\"config\" is a required option for the "
                               "command line.")
          sys.exit(ExitCode.INVALID_OPTION)
        config_dict = vars(config_args)

        for arg_key in command_dict:
          # if args does not contain the arg.
          if arg_key not in command_keys:
            command_dict[arg_key] = config_dict[arg_key]

      # I want to save argument as a namespace..
      args = argparse.Namespace(**command_dict)

      # Checking sanity of the configurations
      if not self.sanity_check(args):
        sys.exit(ExitCode.INVALID_OPTION)

      if is_print_opts:
        self.print_args(args)

      self._args = args
    else:
      self.logger.critical("No options..")
      sys.exit(ExitCode.INVALID_OPTION)

  @staticmethod
  def str2bool(bool_string):
    return bool_string.lower() in ("yes", "true", "t", "1")

  @staticmethod
  def str2list_int(list_string):
    if list_string is None:
      return list_string
    return list(map(int, list_string.replace("\"", "").replace("[", "").
                    replace("]", "").split(",")))

  @property
  def args(self):
    return self._args

  def sanity_check(self, args):
    # Checking sanity of configuration options
    # pylint: disable=too-many-return-statements
    if args.model_caps_type not in ["lowmemory", "einsum", "naive"]:
      self.logger.critical("model-caps-type must be lowmemory, einsum or naive"
                           "but %s", args.model_caps_type)
      return False

    if not args.path_base:
      self.logger.critical("the following arguments are required:"
                           " paths-data-path")
      return False

    if not os.path.isdir(args.path_base)\
        or os.path.isfile(args.path_base):
      self.logger.critical("A data path must exist, please check the data path "
                           "option : %s" % args.path_base)
      return False

    if args.train_schedule_prob is not None and \
      (args.train_schedule_prob < 0 or args.train_schedule_prob >= 2):
      self.logger.critical("Prob. for scheduled sampling must be within [0, 2)"
                           "but %f" % args.train_schdule_prob)
      return False

    if args.train_smoothing_type not in (Constants.SM_LABEL,
                                         Constants.SM_NEIGHBOR):
      self.logger.critical("Please check smoothing type %s" %
                           args.train_smoothing_type)
      return False

    if not args.train_is_mwer and (args.prep_max_inp > 0 or
                                   args.prep_max_tar > 0):
      self.logger.warn("Please do not set max length unless you use mwer, "
                       "but prep-max-inp %d, prep-max-tar %d"%(args.prep_max_inp,
                                                               args.prep_max_tar))

    if args.train_smoothing_type not in (Constants.SM_LABEL,
                                         Constants.SM_NEIGHBOR):
      self.logger.critical("Smoothing type of loss needs to be one of %s, %s" %
                           (Constants.SM_LABEL, Constants.SM_NEIGHBOR))
      return False

    return True

  def print_args(self, args):
    self.logger.info("********************************************")
    self.logger.info("        Settings for Transformer ASR")
    self.logger.info("********************************************")
    sorted_args = sorted(vars(args))
    pre_name = ""
    for arg in sorted_args:
      name = arg.split("_")[0]
      if name != pre_name:
        self.logger.info(". %s"%name.upper())
        pre_name = name

      self.logger.info("- %s=%s"%(arg, getattr(args, arg)))
    self.logger.info("*********************************************")

  @staticmethod
  def build_parser():
    # create parser
    parser = argparse.ArgumentParser(description="Keras based RNN-LM Toolkit ",
                                     fromfile_prefix_chars='@')
    parser.add_argument("--config",
                        help="options can be loaded from this config file")

    # Hyper-parameters for training
    train_group = \
      parser.add_argument_group(title="training",
                                description="Hyper-parameters for model "
                                            "architecture (some of them "
                                            "might be specialized for "
                                            "specific model architecture)")
    train_group.add_argument("--train-inp-dropout", type=float, default=0.1)
    train_group.add_argument("--train-inn-dropout", type=float, default=0.1)
    train_group.add_argument("--train-att-dropout", type=float, default=0.1)
    train_group.add_argument("--train-res-dropout", type=float, default=0.1)
    train_group.add_argument("--train-ckpt-saving-per", type=int, default=1)
    train_group.add_argument('--train-es-min-delta',
                             help="the min delta counting the epoch numbers "
                                  "for earlystop patience", type=float,
                             default=0.001)
    train_group.add_argument('--train-es-tolerance',
                             help="the number of epoch for earlystop patience",
                             type=int, default=1)
    train_group.add_argument('--train-lr-param-k', type=float, default=None,
                             help="learning rate param. defined in stf paper")
    train_group.add_argument('--train-max-epoch', type=int, default=None,
                             help="maximum epoch")
    train_group.add_argument("--train-adam-beta1", type=float, default=0.9)
    train_group.add_argument("--train-adam-beta2", type=float, default=0.98)
    train_group.add_argument("--train-adam-epsilon", type=float, default=1e-09)
    train_group.add_argument("--train-warmup-n", type=int, default=25000)
    train_group.add_argument("--train-ppl-step", type=int, default=1)
    train_group.add_argument("--train-max-step", type=int, default=0)
    train_group.add_argument("--train-opti-type", default=None)
    train_group.add_argument("--train-smoothing-confidence", type=float, default=0.0)
    train_group.add_argument("--train-smoothing-type",
                             default=Constants.SM_NEIGHBOR,
                             help="%s, %s"%(Constants.SM_LABEL,
                                            Constants.SM_NEIGHBOR))
    train_group.add_argument("--train-schedule-prob", type=float, default=None)
    train_group.add_argument("--train-batch-size", type=int, default=26)
    train_group.add_argument('--train-batch-frame', type=int, default=20000)
    train_group.add_argument('--train-lr-max', type=float, default=1e3)
    train_group.add_argument("--train-batch-dynamic", type=ParseOption.str2bool, default="False")
    train_group.add_argument('--train-is-mwer', type=ParseOption.str2bool, default="false")
    train_group.add_argument("--train-batch-buckets",
                             type=ParseOption.str2list_int, default=None)

    # Preprocess
    prep_group = parser.add_argument_group(title="Pre-processing")
    prep_group.add_argument("--prep-data-shard", type=int, default=100)
    prep_group.add_argument("--prep-data-name", default="wsj")
    prep_group.add_argument("--prep-data-unit", default="char")
    prep_group.add_argument("--prep-data-bos", type=ParseOption.str2bool, default="True")
    prep_group.add_argument("--prep-data-pad-space", type=ParseOption.str2bool, default="True")
    prep_group.add_argument('--prep-max-tar', type=int, default=-1)
    prep_group.add_argument('--prep-max-inp', type=int, default=-1)
    prep_group.add_argument('--prep-data-num-train', type=int, default=None)
    prep_group.add_argument('--prep-data-num-valid', type=int, default=None)
    prep_group.add_argument('--prep-data-num-test', type=int, default=None)

    # Path
    path_group = parser.add_argument_group(title="path",
                                           description="paths for inout and "
                                                       "output files")
    path_group.add_argument("--path-base", help="base path")
    path_group.add_argument("--path-ckpt", default=None, help="checkpoint")
    path_group.add_argument("--path-ckpt-epoch", type=int, default=0,
                            help="checkpoint")
    path_group.add_argument("--path-cmvn-ptrn", default=None)
    path_group.add_argument("--path-vocab", help="vocab file")
    path_group.add_argument("--path-hyp", help="recognized text file")
    path_group.add_argument('--path-train-ptrn', default=None)
    path_group.add_argument('--path-test-ptrn', default=None)
    path_group.add_argument('--path-valid-ptrn', default=None)
    path_group.add_argument('--path-train-json', default=None)
    path_group.add_argument('--path-valid-json', default=None)
    path_group.add_argument('--path-test-json', default=None)
    path_group.add_argument('--path-wrt-tfrecord', default=None)

    # Feature
    feature_group = parser.add_argument_group(title="feature",
                                              description="speech feature")
    feature_group.add_argument("--feat-type", default=None, help="stf, stfraw")
    feature_group.add_argument("--feat-dim", type=int, default=None,
                               help="feature dimension")
    feature_group.add_argument("--feat-dim1", type=int, default=None,
                               help="dimension for the first feature")
    feature_group.add_argument("--feat-dim2", type=int, default=None,
                               help="dimension for the second feature")

    # Setting for the entire model
    model_group = parser.add_argument_group(title="model architecture",
                                            description="hyper-parameter for "
                                                        "the entire model")
    model_group.add_argument("--model-encoder-num", type=int, default=None)
    model_group.add_argument("--model-decoder-num", type=int, default=None)
    model_group.add_argument("--model-res-enc", type=int, default=1)
    model_group.add_argument("--model-res-dec", type=int, default=1)
    model_group.add_argument("--model-dimension", type=int, default=1)
    model_group.add_argument("--model-inner-dim", type=int, default=2048)
    model_group.add_argument("--model-inner-num", type=int, default=3)
    model_group.add_argument("--model-att-head-num", type=int, default=4)
    model_group.add_argument("--model-conv-filter-num", type=int, default=64)
    model_group.add_argument("--model-conv-layer-num", type=int, default=2)
    model_group.add_argument("--model-conv-stride", type=int, default=2)
    model_group.add_argument("--model-ckpt-max-to-keep", type=int, default=-1)
    model_group.add_argument("--model-shared-embed", type=ParseOption.str2bool,
                             default="False")
    model_group.add_argument("--model-conv-mask-type", type=int, default=None)
    model_group.add_argument("--model-ap-scale", type=float, default=None)
    model_group.add_argument("--model-ap-width-zero", type=int, default=None)
    model_group.add_argument("--model-ap-width-stripe", type=int, default=None)
    model_group.add_argument("--model-average-num", type=int, default=None)
    model_group.add_argument("--model-ap-encoder", type=ParseOption.str2bool,
                             default="False")
    model_group.add_argument("--model-ap-decoder", type=ParseOption.str2bool,
                             default="False")
    model_group.add_argument("--model-ap-encdec", type=ParseOption.str2bool,
                             default="False")

    model_group.add_argument("--model-type", default="srf")
    model_group.add_argument("--model-initializer", default=None)
    model_group.add_argument("--model-emb-sqrt", type=ParseOption.str2bool,
                             default="True")
    model_group.add_argument("--model-caps-context", type=ParseOption.str2bool,
                             default="True")
    model_group.add_argument("--model-caps-type", default="lowmemory",
                             help="[einsum, lowmemory, naive]")
    model_group.add_argument("--model-caps-iter", type=int, default=2)
    model_group.add_argument("--model-caps-primary-num", type=int, default=3)
    model_group.add_argument("--model-caps-primary-dim", type=int, default=2)
    model_group.add_argument("--model-caps-convolution-num", type=int, default=4)
    model_group.add_argument("--model-caps-convolution-dim", type=int, default=4)
    model_group.add_argument("--model-caps-class-dim", type=int, default=64)
    model_group.add_argument("--model-caps-window-lpad", type=int, default=None)
    model_group.add_argument("--model-caps-window-rpad", type=int, default=None)
    model_group.add_argument("--model-caps-layer-num", type=int, default=2)
    model_group.add_argument("--model-caps-layer-time", type=int, default=None)
    model_group.add_argument("--model-caps-res-connection",
                             type=ParseOption.str2bool, default="False")

    model_group.add_argument("--model-conv-inp-nfilt", type=int, default=64)
    model_group.add_argument("--model-conv-inn-nfilt", type=int, default=128)
    model_group.add_argument("--model-conv-proj-num", type=int, default=3)
    model_group.add_argument("--model-conv-proj-dim", type=int, default=512)

    # Decoding option
    decoding_group = parser.add_argument_group()
    decoding_group.add_argument("--decoding-beam-width", type=int, default=None)
    decoding_group.add_argument("--decoding-lp-alpha", type=float, default=None)
    decoding_group.add_argument("--decoding-from-npy", type=ParseOption.str2bool,
                                default="False")
    return parser

def main():
  logger = Logger(name="data", level=Logger.DEBUG).logger
  ParseOption(sys.argv, logger)

if __name__ == "__main__":
  main()
