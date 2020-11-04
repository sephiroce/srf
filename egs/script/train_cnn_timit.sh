#!/bin/bash

. path.sh

LAYER=${1:-10}
FILT_INP=${2:-128}
FILT_INN=${3:-256}
PROJ_NUM=${4:-3}
PROJ_DIM=${5:-1024}

NAME=CNN_L${LAYER}_NFILT${FILT_INP}_${FILT_INN}_PROJ${PROJ_NUM}_${PROJ_DIM}

function run() {
  SCRIPT=${1}
  K=${2}
  TOLERANCE=${3}
  AVG=${4}
  TC=${5}

  if [ $AVG = "/avg" ]
  then
    MAX_EPOCH=0
    FRAME=1
  else
    MAX_EPOCH=${6}
    AVG=
  fi

  TEST_TFRD=tfrecord_graves13/timit-test-None-123-*-of-*
  if [ $TC = "dev" ]
  then
    TEST_TFRD=tfrecord_graves13/timit-valid-None-123-*-of-*
  fi

  python3.6 -u ${SCRIPT} \
    --path-base=/data/timit \
    --config=egs/conf/timit.conf \
    --path-ckpt=./checkpoint/${NAME}${AVG} \
    --model-type=cnn \
    --model-conv-inp-nfilt=${FILT_INP} \
    --model-conv-inn-nfilt=${FILT_INN} \
    --model-conv-proj-num=${PROJ_NUM} \
    --model-conv-proj-dim=${PROJ_DIM} \
    --train-batch-frame=7000 \
    --train-lr-param-k=${K} \
    --train-es-tolerance=${TOLERANCE} \
    --train-max-epoch=${MAX_EPOCH} \
    --path-test-ptrn=${TEST_TFRD} \
    --train-warmup-n=1200 \
    --model-dimension=1 \
    --model-encoder-num=${LAYER}
}

run tfsr/trainer_sr.py   0.5   27 dummy dummy  27 &>  ${NAME}.1train.out
run tfsr/trainer_sr.py   0.1  200 dummy dummy 200 &>> ${NAME}.1train.out
rm -rf ./checkpoint/${NAME}/avg
run tfsr/utils/average_ckpt_sr.py 1e-6 1 dummy dummy 0 &> ${NAME}.2avg.out
run tfsr/trainer_sr.py   1e-6 0 /avg test 0 &>  ${NAME}.3decode.test.out &
run tfsr/trainer_sr.py   1e-6 0 /avg dev  0 &>  ${NAME}.3decode.valid.out

python3 script/log2utt.py ${NAME}.3decode.test.out > ${NAME}.test.utt
sclite.sh test.ref ${NAME}.test.utt
python3 script/log2utt.py ${NAME}.3decode.valid.out > ${NAME}.valid.utt
sclite.sh valid.ref ${NAME}.valid.utt