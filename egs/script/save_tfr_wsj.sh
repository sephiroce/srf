#!/bin/bash

. path.sh

python3 tfsr/tfsr/data/save_speech_data.py \
--path-base=$DATA_PATH \
--prep-data-shard=100 \
--prep-data-name=wsj \
--path-vocab=egs/data/wsj_31.vocab \
--feat-type=graves13 \
--feat-dim=123 \
--path-train-json=${TRAIN_JSON}.json \
--path-valid-json=${VALID_JSON}.json \
--path-test-json=${TEST_JSON}.json \
--path-wrt-tfrecord=tfrecord_graves \
--prep-data-unit=char \
--decoding-from-npy=True
