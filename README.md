# Supplementary files for the sequential routing framework

## Highlights
. Capsule network only structures can successfully map sequences to sequences
. Mappings are refined by initializing routing iteration based on the previous output
. Sequence-wise routing iteration allows for non-iterative inference
. Structures of capsule network are more important than the number of parameters
. Top layer capsules become similar to the capsule corresponding to a sequence label

## Prerequistes
. Kaldi: https://github.com/kaldi-asr/kaldi  
. SCTK: https://github.com/usnistgov/SCTK/blob/master/doc/sclite.htm  
. Python libraries: Please check requirements.txt in the "src" folder.  

## Directory structure

### manuscript
  - tex source file and figures
### src
  - the python scripts for training and decoding  
### egs
  - result: evaluation results for each table. (Total 24 results)
  - images: 10 figures in png for each diagram  
  - anlaysis: 2 excel files for Fig. 5, 6, and 7.  
  - conf
    > srf.conf: for the sequential routing framework (SRF)  
    > stf.conf: for the speech transformer (STF)  
  - data
    > timit_61.vocab: 62 label symbols. (A blank symbol is automatically added during training and decoding.)  
    > sample.json: input file format for generating TFrecords.  
  - script
    > train_srf.sh: This is a bash script to train and decoding SRF.  
    > train_stf.sh: This is a bash script to train and decoding STF.  
      * Please check "log2utt.py", if you want to see how 61 symbols are mapped to 39 symbols.  

## How to use

### Change ${DATA_PATH} in egs/script/path.sh

### Prepaing TFrecords
  1) Generating wav.scp and text files by running the Kaldi script ${KALDI}/egs/timit/s5/run.sh.  
  2) Extracting features to npy using egs/script/fbank123.sh, then you can find npy files.  
  3) Make json format files by referring to egs/data/sample.json, for examples, train.json, valid.json, and t
est.json.  
  4) run script/save_tfr.sh  

### Change the path in configuations and training scripts.
  - Change the tfrecord file paths in egs/conf/{srf, stf}.conf  
    > --path-train-ptrn=tfrecord_graves13/timit-train-None-123-*-of-*  
    > --path-valid-ptrn=tfrecord_graves13/timit-valid-None-123-*-of-*  
    > --path-test-ptrn=tfrecord_graves13/timit-test-None-123-*-of-*  

### Run scripts to train and evaluate SRF models and STF models.
  - A command line for SRF models  
    > $ egs/script/train_srf.sh $LAYER $PH $CH $DIM $LPAD $RPAD $METHOD $ITER  
      * $LPAD, and $RPAD indicates left and right context size of the window.  
      * $METHOD means routing alogirhtm you can choose SRF or DR.  
      * $ITER is the number of routing iteration.  
  - A command line for STF models
    > $ egs/script/train_stf.sh $LAYER $DIM $INN  
      * $DIM it means the embedding dimension for STF models.  
      * $INN is the dimension of inner layers, i.e. the point-wise feed forward layers in STF models.
