MAX_LENGTH=256
MODEL=/home/whou/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/
OUTPUT_DIR=./temp/
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=64
NUM_EPOCHS=10
# SAVE_STEPS=100
# SAVE_STEPS= $save_steps* gradient_accumulation_steps * batch_size * num_gpus
WARMUP_STEPS=100
SEED=1
LR=3e-5

mkdir -p $OUTPUT_DIR
# python3 -m debugpy --listen 0.0.0.0:8888 --wait-for-client ./run_pl.py \
CUDA_VISIBLE_DEVICES=0 python3 run_pl.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--val_check_interval 100 \
--train_batch_size $TRAIN_BATCH_SIZE \
--eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps 1 \
--learning_rate $LR \
--warmup_steps $WARMUP_STEPS \
--weight_decay 0 \
--seed $SEED \
--do_train \
--do_predict \
--gpus 1 > $OUTPUT_DIR/output.log 2>&1 &
