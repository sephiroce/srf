#!/bin/bash

. path.sh

LAYER=${1:-10}
PH=${2:-60}
CH=${3:-30}
DIM=${4:-20}
LPAD=${5:-2}
RPAD=${6:-2}

NAME=SRF_L${LAYER}_PH${PH}-PD${DIM}-CH${CH}-CD${DIM}-VD${DIM}_W-${LPAD}-${RPAD}

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
    --train-lr-param-k=${K} \
    --train-es-tolerance=${TOLERANCE} \
    --train-max-epoch=${MAX_EPOCH} \
    --path-test-ptrn=${TEST_TFRD} \
    --model-caps-type=lowmemory \
    --model-caps-primary-num=${PH} \
    --model-caps-convolution-num=${CH} \
    --model-caps-primary-dim=${DIM} \
    --model-caps-convolution-dim=${DIM} \
    --model-caps-class-dim=${DIM} \
    --model-caps-window-lpad=${LPAD} \
    --model-caps-window-rpad=${RPAD} \
    --model-caps-context=True \
    --model-caps-iter=1 \
    --model-encoder-num=${LAYER}
}

run tfsr/trainer_sr.py   0.6  15 dummy dummy 15 &>  ${NAME}.1train.out
run tfsr/trainer_sr.py   0.5  50 dummy dummy 50 &>  ${NAME}.1train.out
run tfsr/trainer_sr.py   0.1  70 dummy dummy 70 &>  ${NAME}.1train.out
run tfsr/trainer_sr.py   0.05 80 dummy dummy 80 &>> ${NAME}.1train.out
rm -rf ./checkpoint/${NAME}/avg
run tfsr/utils/average_ckpt_sr.py 1e-6 1 dummy dummy 0 &> ${NAME}.2avg.out
run tfsr/trainer_sr.py   1e-6 0 /avg test 0 &>  ${NAME}.3decode.test.out &
run tfsr/trainer_sr.py   1e-6 0 /avg dev  0 &>  ${NAME}.3decode.valid.out

python3 tfsr/utils/log2utt_wsj.py ${NAME}.3decode.test.out > ${NAME}.test.utt
egs/script/sclite.sh test_wsj.ref ${NAME}.test.utt
python3 tfsr/utils/log2utt_wsj.py ${NAME}.3decode.valid.out > ${NAME}.valid.utt
egs/script/sclite.sh valid_wsj.ref ${NAME}.valid.utt
