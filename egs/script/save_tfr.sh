#!/bin/bash

. path.sh

python3 tfsr/tfsr/data/save_speech_data.py \
--path-base=$DATA_PATH \
--prep-data-shard=10 \
--prep-data-name=timit \
--path-vocab=egs/data/timit_61.vocab \
--feat-type=graves13 \
--feat-dim=123 \
--path-train-json=train_61.json \
--path-valid-json=valid_61.json \
--path-test-json=test_61.json \
--path-wrt-tfrecord=tfrecord_graves \
--prep-data-unit=word \
--decoding-from-npy=True
