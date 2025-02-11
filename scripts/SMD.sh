export CUDA_VISIBLE_DEVICES=4

dataPath="./dataset/SMD/processed"
savePath="./checkpoints/SMD"

python main.py --anormly_ratio 0.5 --num_epochs 20   --batch_size 256  --mode train --dataset SMD  --data_path ${dataPath} --input_c 38 --model_save_path ${savePath}
python main.py --anormly_ratio 0.5 --num_epochs 20   --batch_size 256  --mode test  --dataset SMD  --data_path ${dataPath} --input_c 38 --pretrained_model 20 --model_save_path ${savePath}
