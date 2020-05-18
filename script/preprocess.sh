#!/usr/bin/env bash

python -u preprocess.py \
        -train_dir=../data/distractor/race_train_updated.json \
        -valid_dir=../data/distractor/race_dev_updated.json \
        -save_data=../data/processed_300_50_30 \
        -share_vocab \
        -total_token_length=300 \
        -src_seq_length=50 \
        -src_sent_length=30 \
        -lower

python -u embeddings_to_torch.py \
        -emb_file_enc=../data/glove.840B.300d.txt \
        -emb_file_dec=../data/glove.840B.300d.txt \
        -output_file=../data/processed_300_50_30.glove \
        -dict_file=../data/processed_300_50_30.vocab.pt
