#!/bin/bash

cleanup() {
    echo "Terminating all Python scripts..."
    # 终止所有子进程
    pkill -P $$
    exit 1
}
trap cleanup SIGINT SIGTERM


CUDA_VISIBLE_DEVICES=1 python src/pgm.py --dataset "BACE"    --run_name "pgm-BACE-001"    --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=2 python src/pgm.py --dataset "HIV"     --run_name "pgm-HIV-001"     --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=3 python src/pgm.py --dataset "MUV"     --run_name "pgm-MUV-001"     --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=1 python src/pgm.py --dataset "PCBA"    --run_name "pgm-PCBA-001"    --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=2 python src/pgm.py --dataset "BBBP"    --run_name "pgm-BBBP-001"    --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=3 python src/pgm.py --dataset "ClinTox" --run_name "pgm-ClinTox-001" --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=1 python src/pgm.py --dataset "SIDER"   --run_name "pgm-SIDER-001"   --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=2 python src/pgm.py --dataset "Tox21"   --run_name "pgm-Tox21-001"   --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=3 python src/pgm.py --dataset "ToxCast" --run_name "pgm-ToxCast-001" --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &


CUDA_VISIBLE_DEVICES=1 python src/pgm.py --dataset "ESOL"          --run_name "pgm-ESOL-001"          --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=2 python src/pgm.py --dataset "FreeSolv"      --run_name "pgm-FreeSolv-001"      --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &
CUDA_VISIBLE_DEVICES=3 python src/pgm.py --dataset "Lipophilicity" --run_name "pgm-Lipophilicity-001" --seed 666 --num-epochs 10 --batch_size 32 --learning_rate 0.001 --save_ckpt_step 1 --log_path ./log --split scaffold --split-ratio "0.8,0.1,0.1" --num-workers 0 --wandb &


wait
echo "All Python scripts have finished."