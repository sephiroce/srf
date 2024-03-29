#!/bin/bash

. path.sh

LAYER=${1:-15}
FILT_INP=${2:-200}
FILT_INN=${3:-430}
PROJ_NUM=${4:-3}
PROJ_DIM=${5:-2048}
STRIDE=${6:-2}
IS_MP=${7:-False}

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

  TEST_TFRD=tfrecord_graves13/wsj-test-graves13-123-*-of-*
  if [ $TC = "dev" ]
  then
    TEST_TFRD=tfrecord_graves13/wsj-valid-graves13-123-*-of-*
  fi

  python3.6 -u ${SCRIPT} \
    --path-base=/data/wsj \
    --config=egs/conf/wsj.conf \
    --path-ckpt=./checkpoint/${NAME}${AVG} \
    --model-type=cnn \
    --model-conv-inp-nfilt=${FILT_INP} \
    --model-conv-inn-nfilt=${FILT_INN} \
    --model-conv-proj-num=${PROJ_NUM} \
    --model-conv-proj-dim=${PROJ_DIM} \
    --model-conv-stride=${STRIDE} \
    --train-lr-param-k=${K} \
    --model-conv-is-mp=${IS_MP} \
    --train-es-tolerance=${TOLERANCE} \
    --train-max-epoch=${MAX_EPOCH} \
    --path-test-ptrn=${TEST_TFRD} \
    --model-dimension=1 \
    --model-encoder-num=${LAYER}
}

run tfsr/trainer_sr.py   0.2  15 dummy dummy 15 &>  ${NAME}.1train.out
run tfsr/trainer_sr.py   0.08 50 dummy dummy 50 &>> ${NAME}.1train.out
run tfsr/trainer_sr.py   0.04 70 dummy dummy 70 &>> ${NAME}.1train.out
run tfsr/trainer_sr.py   0.02 80 dummy dummy 80 &>> ${NAME}.1train.out
rm -rf ./checkpoint/${NAME}/avg
run tfsr/utils/average_ckpt_sr.py 1e-6 1 dummy dummy 0 &> ${NAME}.2avg.out
run tfsr/trainer_sr.py   1e-6 0 /avg test 0 &>  ${NAME}.3decode.test.out &
run tfsr/trainer_sr.py   1e-6 0 /avg dev  0 &>  ${NAME}.3decode.valid.out

python3 tfsr/utils/log2utt_wsj.py ${NAME}.3decode.test.out > ${NAME}.test.utt
egs/script/sclite.sh test_wsj.ref ${NAME}.test.utt
python3 tfsr/utils/log2utt_wsj.py ${NAME}.3decode.valid.out > ${NAME}.valid.utt
egs/script/sclite.sh valid_wsj.ref ${NAME}.valid.utt
