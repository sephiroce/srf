#!/bin/bash

. path.sh

python3 tfsr/tfsr/data/save_speech_data.py \
--path-base=$DATA_PATH \
--prep-data-shard=10 \
--prep-data-name=timit \
--path-vocab=egs/data/timit_62.vocab \
--feat-type=graves13 \
--feat-dim=123 \
--path-train-json=${TRAIN_JSON}.json \
--path-valid-json=${VALID_JSON}.json \
--path-test-json=${TEST_JSON}.json \
--path-wrt-tfrecord=tfrecord_graves \
--prep-data-unit=word \
--decoding-from-npy=True
