# Supplementary files for the sequential routing framework
* Arxiv: https://arxiv.org/abs/2007.11747
* Author: Kyungmin Lee (sephiroce@snu.ac.kr)

## Highlights
  - Capsule network only structures can successfully map sequences to sequences
  - Mappings are refined by initializing routing iteration based on the previous output
  - Sequence-wise routing iteration allows for non-iterative inference
  - Structures of capsule network are more important than the number of parameters
  - Top layer capsules become similar to the capsule corresponding to a sequence label

## Prerequisites
  - Tensorflow >= 2.3.0
  - Cuda >= 10.0
  - Kaldi: https://github.com/kaldi-asr/kaldi
  - SCTK: https://github.com/usnistgov/SCTK/blob/master/doc/sclite.htm
  - Python libraries: Please check requirements.txt in the "tfsr" folder.
    - "tfsr" stands for "**T**ensor**F**low based **S**peech **R**ecognition toolkit"

## Directory structure
### tfsr
  - the python scripts for training and decoding
### egs
  - result: 1 excel file for Table 1-5 and related logs.
  - images: 10 figures in jpg for each diagram  
  - analysis: 2 excel files for Fig. 5, 6, and 7.  
  - conf
    - {timit, wsj}.conf: a configuration for the TIMIT and WSJ corpus  
  - data
    - timit_62.vocab: 62 label symbols.  
    - wsj_31.vocab: 62 label symbols.    
      * A blank symbol is automatically added during training and decoding.
    - sample.json: input file format for generating TFrecords.  
  - script
    > train_{srf, cnn, lstm, stf}_{timit, wsj}.sh: This is a bash script to train and decoding models.  
      * Please check "log2utt.py", if you want to see how 61 symbols are mapped to 39 symbols for the TIMIT corpus.  

## How to use
### Preparing TFrecords
  1) Generating wav.scp and text files by running the Kaldi script ${KALDI}/egs/timit/s5/run.sh or ${KALDI}/egs/wsj/s5/run.sh   
  2) Extracting features to npy using egs/script/fbank123.sh, then you can find npy files.  
  3) Make json format files by referring to egs/data/sample.json, for examples, train.json, valid.json, and test.json.  
  4) run script/save_tfr.sh  

### Change the path in configurations and training scripts.
  - egs/conf/{timit, wsj}.conf
    - path-{train, valid, test}-ptrn: file patterns of TFrecords
  - training/decoding scripts
    - path-base: base path for TFrecords, vocab files and configuration files.   

### Run scripts to train and evaluate
  - Sequential Routing Framework (SRF)  
    ```$egs/script/train_srf_{timit, wsj}.sh $LAYER $PH $CH $DIM $LPAD $RPAD $METHOD $ITER```
    * $PH and $CH: heights of primary and convolutional capsule groups
    * $DIM: the depth of all capsule groups    
    * $LPAD, and $RPAD indicates left and right context size of the window.  
    * $METHOD means the routing algorithm you can choose SDR or DR.  
    * $ITER is the number of routing iteration.  
  - Speech TransFormer (STF)-based CTC network  
    ```$egs/script/train_stf_{timit,wsj}.sh $LAYER $DIM $INN```  
    * $DIM means the embedding dimension for STF models.  
    * $INN is the dimension of inner layers, i.e. the point-wise feed forward layers in STF models.
  - Bi/Uni-directional Long Short Term Memory-based CTC network  
    ```$egs/script/train_lstm_wsj.sh $LAYER $TYPE $DIM```
    * $TYPE: blstm or ulstm
    * $DIM means cell sizes
  - Convolutional Neural Network (CNN)-based CTC network  
    ```$egs/script/train_cnn_{timit,wsj}.sh $LAYER $FILT_INP $FILT_INN $PROJ_NUM $PROJ_DIM```
    * $FILT_INP: the number of filters for the first four layers
    * $FILT_INN: the number of filters for the rest of layers  
    * $PROJ_NUM: the number of feed forwarding layers
    * $PROJ_DIM: the number of neurons in feed forwarding layers   
