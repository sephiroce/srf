#!/bin/bash

. path.sh

LAYER=${1:-20}
DIM=${2:-256}
INN=${3:-1488}

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

  TEST_TFRD=tfrecord_graves13/wsj-test-graves13-123-*-of-*
  if [ $TC = "dev" ]
  then
    TEST_TFRD=tfrecord_graves13/wsj-valid-graves13-123-*-of-*
  fi

  python3.6 -u ${SCRIPT} \
    --path-base=/data/wsj \
    --config=egs/conf/wsj.conf \
    --path-ckpt=./checkpoint/${NAME}${AVG} \
    --model-inner-dim=${INN} \
    --train-att-dropout=0.3 \
    --train-inn-dropout=0.4 \
    --train-inp-dropout=0.3 \
    --train-res-dropout=0.4 \
    --model-ap-scale=1 \
    --model-ap-encoder=True \
    --model-ap-width-zero=1 \
    --model-ap-width-stripe=1 \
    --model-ap-scale=1 \
    --model-ap-encoder=True \
    --model-ap-decoder=True \
    --model-ap-encdec=False \
    --model-dimension=${DIM} \
    --train-lr-param-k=${K} \
    --train-es-tolerance=${TOLERANCE} \
    --train-max-epoch=${MAX_EPOCH} \
    --path-test-ptrn=${TEST_TFRD} \
    --model-encoder-num=${LAYER}
}

run tfsr/trainer_tf.py 10  27 dummy dummy 27 &>  ${NAME}.1train.out
run tfsr/trainer_tf.py  1  70 dummy dummy 70 &>> ${NAME}.1train.out
run tfsr/trainer_tf.py 0.5 80 dummy dummy 80 &>> ${NAME}.1train.out
rm -rf ./checkpoint/${NAME}/avg
run tfsr/utils/average_ckpt_tf.py 1e-6 1 dummy dummy 0 &>  ${NAME}.2avg.out
run tfsr/trainer_tf.py   1e-6 0 /avg test 0 &>  ${NAME}.3decode.test.out &
run tfsr/trainer_tf.py   1e-6 0 /avg dev  0 &>  ${NAME}.3decode.valid.out

python3 tfsr/utils/log2utt_wsj.py ${NAME}.3decode.test.out > ${NAME}.test.utt
egs/script/sclite.sh test_wsj.ref ${NAME}.test.utt
python3 tfsr/utils/log2utt_wsj.py ${NAME}.3decode.valid.out > ${NAME}.valid.utt
egs/script/sclite.sh valid_wsj.ref ${NAME}.valid.utt
