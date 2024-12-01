#!/bin/bash

cleanup() {
    echo "Terminating all Python scripts..."
    # 终止所有子进程
    pkill -P $$
    exit 1
}
trap cleanup SIGINT SIGTERM


# CUDA_VISIBLE_DEVICES=1 python src/pretrain.py --dataset "BACE"    --run_name "pretrain-BACE-001"    --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=2 python src/pretrain.py --dataset "HIV"     --run_name "pretrain-HIV-001"     --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
# CUDA_VISIBLE_DEVICES=2 python src/pretrain.py --dataset "MUV"     --run_name "pretrain-MUV-001"     --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=3 python src/pretrain.py --dataset "PCBA"    --run_name "pretrain-PCBA-001"    --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
# CUDA_VISIBLE_DEVICES=3 python src/pretrain.py --dataset "BBBP"    --run_name "pretrain-BBBP-001"    --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
# CUDA_VISIBLE_DEVICES=1 python src/pretrain.py --dataset "ClinTox" --run_name "pretrain-ClinTox-001" --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
# CUDA_VISIBLE_DEVICES=2 python src/pretrain.py --dataset "SIDER"   --run_name "pretrain-SIDER-001"   --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
# CUDA_VISIBLE_DEVICES=3 python src/pretrain.py --dataset "Tox21"   --run_name "pretrain-Tox21-001"   --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
# CUDA_VISIBLE_DEVICES=1 python src/pretrain.py --dataset "ToxCast" --run_name "pretrain-ToxCast-001" --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &


# CUDA_VISIBLE_DEVICES=2 python src/pretrain.py --dataset "ESOL"          --run_name "pretrain-ESOL-001"          --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
# CUDA_VISIBLE_DEVICES=3 python src/pretrain.py --dataset "FreeSolv"      --run_name "pretrain-FreeSolv-001"      --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
# CUDA_VISIBLE_DEVICES=1 python src/pretrain.py --dataset "Lipophilicity" --run_name "pretrain-Lipophilicity-001" --seed 666 --num-epochs 200 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &


wait
echo "All Python scripts have finished."