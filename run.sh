#!/bin/bash

# 设置超参数
EMB_SIZE=300
HID_SIZE=150
EPOCHS=500
CLIP_GRAD=1
LR=1e-5
MOMENTUM=0.9
B_SIZE=4
SAVE="saved_model/model_3.pth"
LOAD="none"
LOG_FILE="saved_model/log_3.txt"
TRAIN_DATA="dataset/documents_train.json"
TEST_DATA="dataset/documents_test.json"
ENT_VEC="dataset/ent_vec.txt"
WORD_INFO="dataset/word_info.txt"
STOP_WORD="dataset/stopword.txt"

# 运行 main.py 并传递超参数
python main.py \
    --emb_size $EMB_SIZE \
    --hid_size $HID_SIZE \
    --epochs $EPOCHS \
    --clip_grad $CLIP_GRAD \
    --lr $LR \
    --momentum $MOMENTUM \
    --b_size $B_SIZE \
    --save $SAVE \
    --load $LOAD \
    --log_file $LOG_FILE \
    --train_data $TRAIN_DATA \
    --test_data $TEST_DATA \
    --ent_vec $ENT_VEC \
    --word_info $WORD_INFO \
    --stop_word $STOP_WORD
