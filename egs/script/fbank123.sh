#!/bin/bash
#0. prep using kaldi script
#spk2utt, utt2spk
#wav.scp

export PATH=${KALDI_HOME}/src/featbin:${KALDI_HOME}/bin:$PATH

#1. fbank for all utterance + utt2spk
compute-fbank-feats \
--num-mel-bins=40 \
--sample-frequency=16000 \
--use-log-fbank=True \
--use-energy=True \
scp:wav.scp ark:- | add-deltas ark:- ark,scp:feats.ark,feats.scp

#2. cmvn using fbank feats and utt2spk
compute-cmvn-stats --spk2utt=ark:spk2utt scp:feats.scp ark,scp:cmvn.ark,cmvn.scp
apply-cmvn --utt2spk=ark:utt2spk scp:cmvn.scp scp:feats.scp ark,scp:normed_feats.ark,normed_feat.scp

#3. save cmvn applied fbanks to npy
copy-feats scp:normed_feat.scp ark,t:normed_feats.txt
python3 parsing.py normed_feats.txt
