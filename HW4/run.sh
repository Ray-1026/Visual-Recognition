export CUDA_VISIBLE_DEVICES=1

EXP_NAME=exp
CKPT_DIR=ckpts/$EXP_NAME
LOG_DIR=$EXP_NAME

# train & test
python train.py --log_dir $LOG_DIR --save_ckpt_dir $CKPT_DIR --batch_size 4 --epochs 150 --fp16
python test.py --test_only --ckpt_path $CKPT_DIR/epoch=149-step=120000.ckpt