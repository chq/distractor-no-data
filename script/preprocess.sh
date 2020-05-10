#!/usr/bin/env bash

python -u preprocess.py \
        -train_dir=../data/distractor/race_train_updated.json \
        -valid_dir=../data/distractor/race_dev_updated.json \
        -save_data=../data/processed \
        -share_vocab \
        -total_token_length=500 \
        -src_seq_length=60 \
        -src_sent_length=40 \
        -lower

python embeddings_to_torch.py \
        -emb_file_enc=../data/glove.840B.300d.txt \
        -emb_file_dec=../data/glove.840B.300d.txt \
        -output_file=../data/processed.glove \
        -dict_file=../data/processed.vocab.pt
