# 配置路径
TRAIN_CSV=./dataset/train_seed_42_split_80.csv
TEST_CSV=./dataset/test_seed_42_split_20.csv
OUTPUT_DIR=./output/seed_42_split_80

# 模型配置
BERT_MODEL="models--hfl--chinese-bert-wwm-ext"
IMAGE_MODEL="models--facebook--convnextv2-huge-22k-512"


BATCH_SIZE=16
EPOCHS=10
LR=1e-3
LR_BERT=2e-5
LR_IMAGE=5e-6

TARGET_IMAGE_SIZE=512  # Huge 模型输入是 512x512

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=3 uv run train.py \
    --train-csv ${TRAIN_CSV} \
    --test-csv ${TEST_CSV} \
    --output-dir ${OUTPUT_DIR} \
    --bert-model ${BERT_MODEL} \
    --image-model ${IMAGE_MODEL} \
    --max-seq-len 128 \
    --target-image-size ${TARGET_IMAGE_SIZE} \
    --resize-mode padding \
    --num-transformer-layers 4 \
    --num-transformer-heads 4 \
    --dropout 0.2 \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --lr-bert ${LR_BERT} \
    --lr-image ${LR_IMAGE} \
    --weight-decay 0.01 \
    --num-workers 4 \
    --use-amp \
    --save-every 1 \
    --accumulation-steps 4 \
    --seed 42