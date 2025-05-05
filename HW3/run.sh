export CUDA_VISIBLE_DEVICES=0

EXP_NAME=exp
CKPT_DIR=ckpts/$EXP_NAME
LOG_DIR=$EXP_NAME

# train
python main.py --log_dir $LOG_DIR --save_ckpt_dir $CKPT_DIR --epochs 20 --fp16 --backbone c \
                --lr 1e-4 --optimizer adamw --weight_decay 1e-2 --scheduler cosine

# test
python main.py --test_only --ckpt_name $CKPT_DIR/last.pth --backbone c