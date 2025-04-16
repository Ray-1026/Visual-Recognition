export CUDA_VISIBLE_DEVICES=0

EXP_NAME=exp
CKPT_DIR=ckpts/$EXP_NAME
LOG_DIR=$EXP_NAME
CKPT_NAME=ckpts/$EXP_NAME/best.pth
python main.py --log_dir $LOG_DIR --save_ckpt_dir $CKPT_DIR --epochs 10 --fp16 --scheduler cosine --weight_decay 1e-2
python main.py --test_only --ckpt_name $CKPT_NAME --box_score_thresh 0.7