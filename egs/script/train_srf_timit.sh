#!/bin/bash

. path.sh

LAYER=${1:-7}
PH=${2:-60}
CH=${3:-30}
DIM=${4:-8}
LPAD=${5:-1}
RPAD=${6:-1}
METHOD=${7:-"SDR"}
ITER=${8:-1}

if [ ${METHOD} = "DR" ]
then
  ROUTING="false"
else
  ROUTING="true"
fi

NAME=SRF_L${LAYER}_PH${PH}-PD${DIM}-CH${CH}-CD${DIM}-VD${DIM}_W-${LPAD}-${RPAD}_${METHOD}-I${ITER}

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
    --train-lr-param-k=${K} \
    --train-batch-frame=7000 \
    --train-es-tolerance=${TOLERANCE} \
    --train-max-epoch=${MAX_EPOCH} \
    --path-test-ptrn=${TEST_TFRD} \
    --model-caps-primary-num=${PH} \
    --model-caps-convolution-num=${CH} \
    --model-caps-primary-dim=${DIM} \
    --model-caps-convolution-dim=${DIM} \
    --model-caps-class-dim=${DIM} \
    --model-caps-type=naive \
    --model-caps-window-lpad=${LPAD} \
    --model-caps-window-rpad=${RPAD} \
    --model-caps-context=${ROUTING} \
    --model-caps-iter=${ITER} \
    --model-encoder-num=${LAYER}
}

run tfsr/trainer_sr.py 0.5  27 dummy dummy  27 &>  ${NAME}.1train.out
run tfsr/trainer_sr.py 0.1 200 dummy dummy 200 &>> ${NAME}.1train.out
rm -rf ./checkpoint/${NAME}/avg
run tfsr/utils/average_ckpt_sr.py 1e-6 1 dummy dummy 0 &> ${NAME}.2avg.out
run tfsr/trainer_sr.py 1e-6 0 /avg test 0 &> ${NAME}.3decode.test.out &
run tfsr/trainer_sr.py 1e-6 0 /avg dev  0 &> ${NAME}.3decode.valid.out

python3 tfsr/utils/log2utt.py ${NAME}.3decode.test.out > ${NAME}.test.utt
egs/script/sclite.sh test.ref ${NAME}.test.utt
python3 tfsr/utils/log2utt.py ${NAME}.3decode.valid.out > ${NAME}.valid.utt
egs/script/sclite.sh valid.ref ${NAME}.valid.utt