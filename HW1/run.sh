export CUDA_VISIBLE_DEVICES=0

python main.py --model resnest101 --num_epochs 50 --log_dir renest101_tri
python main.py --model resnest50 --num_epochs 50 --log_dir renest50_tri
python main.py --model senext50 --num_epochs 80 --log_dir senext50_tri
python ensemble.py
