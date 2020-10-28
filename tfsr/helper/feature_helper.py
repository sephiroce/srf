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

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"


import numpy as np
from scipy.io import wavfile as wav
from python_speech_features import mfcc, fbank, logfbank
from python_speech_features.base import delta


def get_feats(wav_path, feat_type):
  feats = None
  feature_file_name = wav_path + "." + feat_type
  try:
    feats = np.load(feature_file_name + ".npy")
  except:
    if feat_type == "stf":
      feats = feat_speech_transformer(wav_path, 1, True)
    elif feat_type == "stf2":
      feats = feat_speech_transformer(wav_path, 2, True)
    elif feat_type == "stf3":
      feats = feat_speech_transformer(wav_path, 3, True)
    elif feat_type == "stfraw":
      feats = feat_speech_transformer_raw(wav_path, 1)
    elif feat_type == "raw":
      feats = raw_speech(wav_path)
    elif feat_type == "raw80":
      feats = raw_speech(wav_path, 80)
    elif feat_type == "fb1":
      feats = feat_speech_transformer(wav_path, 1, False)
    elif feat_type == "fb2":
      feats = feat_speech_transformer(wav_path, 2, False)
    elif feat_type == "las":
      feats = feat_listen_and_spell(wav_path, 40)
    elif feat_type == "lasraw":
      feats = feat_listen_and_spell_raw(wav_path, 40)
    elif feat_type == "fbank80":
      feats = feat_listen_and_spell(wav_path, 80)
    elif feat_type == "fbank80raw":
      feats = feat_listen_and_spell_raw(wav_path, 80)
    elif feat_type == "spectrogram":
      feats = spectrogram(wav_path)
    elif feat_type == "graves13":
      feats = graves_2013(wav_path)

    np.save(feature_file_name, feats)

  return feats


def spectrogram(wav_path):
  (rate, sig) = wav.read(wav_path)
  from scipy import signal
  _, _, spectrogram = signal.spectrogram(sig, rate, nperseg=158) # 80
  return np.transpose(spectrogram)


def feat_listen_and_spell_raw(wav_path, nfilt=40):
  fbank_feat = feat_listen_and_spell(wav_path, nfilt)
  raw_feat = raw_speech(wav_path)
  diff = (np.shape(raw_feat)[0] - np.shape(fbank_feat)[0])
  start = int(diff / 2)
  end = int(np.shape(raw_feat)[0] - diff / 2)
  return np.concatenate((fbank_feat, raw_feat[start:end, :]), axis=1)


def feat_listen_and_spell(wav_path, nfilt=40):
  """
  William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals
  Listen, Attend and Spell
  ICASSP 2016

  40 - dimensional logmel filter bank features were computed every 10 ms and
  used as the acoustic inputs to the listener.
  """
  (rate, sig) = wav.read(wav_path)
  fbank_feat = logfbank(signal=sig, samplerate=rate, nfilt=nfilt)
  return fbank_feat


def feat_speech_transformer_raw(wav_path, delta_window_width):
  """
  :return:
  """
  fbank_feat = feat_speech_transformer(wav_path, delta_window_width, True)
  raw_feat = raw_speech(wav_path)
  diff = (np.shape(raw_feat)[0] - np.shape(fbank_feat)[0])
  start = int(diff / 2)
  end = int(np.shape(raw_feat)[0] - diff / 2)
  return np.concatenate((fbank_feat, raw_feat[start:end, :]), axis=1)


def feat_speech_transformer(wav_path, delta_window_width, is_log):
  """
  Linhao Dong, Shuang Xu, Bo Xu
  SPEECH-TRANSFORMER: A NO-RECURRENCE SEQUENCE-TO-SEQUENCE MODEL FOR SPEECH RECOGNITION
  ICASSP 2018

  Filterbank features
  The input acoustic features were 80-dimensional filterbanks extracted with a
  hop size of 10ms and a window size of 25ms, extended with temporal first and
  second order differences.

  CMVN
  and per-speaker mean subtraction and variance normalization.

  :return:
  """
  (rate, sig) = wav.read(wav_path)

  # computing features
  if is_log:
    fbank_feat = logfbank(signal=sig, samplerate=rate, nfilt=80)
  else:
    fbank_feat, _ = fbank(signal=sig, samplerate=rate, nfilt=80)

  # concatenating delta vectors
  delta_feat = delta(fbank_feat, delta_window_width)
  delta_delta_feat = delta(delta_feat, delta_window_width)
  return np.concatenate((fbank_feat, delta_feat, delta_delta_feat), axis=1)


def raw_speech(wav_path, dim=160):
  """
  Zolt´an T¨uske, Pavel Golik, RalfSchl¨uter, Hermann Ney
  Acoustic Modeling with Deep Neural Networks Using Raw Time Signal for LVCSR
  Interspeech 2014

  2.1. Waveform — "raw" time signal
  Processing the audio sampled at 16 kHz with the same 10 ms steps as in case of
  typical cepstral features boils down to taking 160 samples from the PCM
  waveform. The windows are non- overlapping so that stacking neighboring
  vectors does not result in discontinuities. The samples quantized with 16 bit
  need to be normalized to a numerically robust range by performing the mean and
  variance normalization either globally over the complete training data or on
  the per-utterance level. This can be interpreted as DC bias removal and
  loudness equalization and at the same time it serves numerical purposes to
  stabilize the DNN training with gradient descent.

  :return:
  """
  (_, sig) = wav.read(wav_path)
  sig = np.pad(sig, (0, dim - len(sig) % dim), mode='constant')
  return np.array(sig).reshape([-1, dim])


def graves_2012(wav_path):
  """
  Alex. Graves:
  Sequence Transduction with Recurrent Neural Networks.
  CoRR abs/1211.3711 (2012)

  MFCC features
  Standard speech preprocessing was applied to transform the audio files into
  feature sequences. 26 channel mel-frequency filter bank and a pre-emphasis
  coefficient of 0.97 were used to compute 12 mel-frequency cepstral coeffici-
  ents plus an energy coefficient on 25ms Hamming windows at 10ms intervals.
  Delta coefficients were added to create input sequences of length 26 vectors

  For CMVN
  and all coefficient were normalised to have mean zero and standard deviat-
  ion one over the train- ing set. ==> please set --prep-cmvn-samples to -1.

  I left as default the other options which were not mentioned in the paper
  such as nfft, lowfreq, highfreq, ceplifter, etc.

  :param wav_path: wav file path
  :return: a feature sequence
  """
  (rate, sig) = wav.read(wav_path)
  # computing features
  mfcc_feat = \
    mfcc(signal=sig, samplerate=rate, numcep=12, winlen=0.025, nfilt=26,
         winstep=0.01, preemph=0.97, appendEnergy=False, winfunc=np.hamming)
  # adding energy
  energy = np.expand_dims(np.sum(np.power(mfcc_feat, 2), axis=-1), 1)
  mfcc_e_feat = np.concatenate((energy, mfcc_feat), axis=-1)
  # concatenating a delta vector
  delta_feat = delta(mfcc_e_feat, 1)
  return np.concatenate((mfcc_e_feat, delta_feat), axis=1)


def graves_2013(wav_path):
  """
  Alex Graves, Abdel-rahman Mohamed, Geoffrey E. Hinton:
  Speech recognition with deep recurrent neural networks.
  ICASSP 2013: 6645-6649

  FBANK features : (40 fbank, 1 energy * 3)
  The audio data was encoded using a Fourier-transform-based filter-bank with
  40 coefficients (plus energy) distributed on a mel-scale, together with their
  first and second temporal derivatives. Each input vector was therefore size 123.

  For CMVN
  The data were normalised so that every element of the input vec- tors had
  zero mean and unit variance over the training set.

  there is not description about window I chose to use a hanning window.

  I left as default the other options which were not mentioned in the paper
  such as nfft, lowfreq, highfreq, ceplifter, etc.

  :param wav_path: wav file path
  :return: a feature sequence
  """
  (rate, sig) = wav.read(wav_path)
  # computing features
  fbank_feat, _ = \
    fbank(signal=sig, samplerate=rate, nfilt=40, winfunc=np.hanning)

  # adding energy
  energy = np.expand_dims(np.sum(np.power(fbank_feat, 2), axis=-1), 1)
  fbank_e_feat = np.concatenate((energy, fbank_feat), axis=-1)
  # concatenating delta vectors
  delta_feat = delta(fbank_e_feat, 2)
  delta_delta_feat = delta(delta_feat, 2)
  return np.concatenate((fbank_e_feat, delta_feat, delta_delta_feat), axis=1)
