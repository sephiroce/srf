#!/bin/bash

. path.sh

LAYER=${1:-20}
DIM=${2:-128}
INN=${3:-1024}

NAME=TF_L${LAYER}_D${DIM}_H${INN}

function run() {
  SCRIPT=${1}
  K=${2}
  TOLERANCE=${3}
  AVG=${4}
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
    --model-inner-dim=${INN} \
    --train-att-dropout=0.3 \
    --train-inn-dropout=0.4 \
    --train-inp-dropout=0.3 \
    --train-res-dropout=0.4 \
    --model-dimension=${DIM} \
    --train-warmup-n=1000 \
    --train-batch-frame=20000 \
    --train-lr-param-k=${K} \
    --train-es-tolerance=${TOLERANCE} \
    --train-max-epoch=${MAX_EPOCH} \
    --path-test-ptrn=${TEST_TFRD} \
    --model-encoder-num=${LAYER}
}

run tfsr/trainer_tf.py 1.5  27 dummy dummy  27 &>  ${NAME}.1train.out
run tfsr/trainer_tf.py 0.5 200 dummy dummy 200 &>> ${NAME}.1train.out
rm -rf ./checkpoint/${NAME}/avg
run tfsr/utils/average_ckpt_tf.py 1e-6 1 dummy dummy 0 &>  ${NAME}.2avg.out
run tfsr/trainer_tf.py   1e-6 0 /avg test 0 &>  ${NAME}.3decode.test.out &
run tfsr/trainer_tf.py   1e-6 0 /avg dev  0 &>  ${NAME}.3decode.valid.out

python3 tfsr/utils/log2utt.py ${NAME}.3decode.test.out > ${NAME}.test.utt
egs/script/sclite.sh test.ref ${NAME}.test.utt
python3 tfsr/utils/log2utt.py ${NAME}.3decode.valid.out > ${NAME}.valid.utt
egs/script/sclite.sh valid.ref ${NAME}.valid.utt
