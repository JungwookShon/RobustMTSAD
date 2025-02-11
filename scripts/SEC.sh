export CUDA_VISIBLE_DEVICES=7

dataPath="./dataset/processed"
savePath="./checkpoints/SEC"

python main.py --anormly_ratio 0.022 --num_epochs 20   --batch_size 128  --mode train --dataset SEC  --data_path ${dataPath} --input_c 13 --output_c 13 --model_save_path ${savePath} --win_size 100
python main.py --anormly_ratio 0.022 --num_epochs 20   --batch_size 128  --mode test  --dataset SEC  --data_path ${dataPath} --input_c 13 --output_c 13 --pretrained_model 20 --model_save_path ${savePath} --win_size 100
