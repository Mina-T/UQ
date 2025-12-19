#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --output=out
#SBATCH --open-mode=append
#SBATCH --mem=120000MB
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=mace
#SBATCH --time 23:50:00

touch $SLURM_JOB_ID
date

python3 run_train.py \
    --name="MACE_model" \
    --train_file=$train_data_dir \
    --valid_file=$valid_data_dir \
    --test_file=$test_data_dir \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{6:-5.5}' \
    --atomic_numbers="[6]" \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=6.0 \
    --batch_size=20 \
    --max_num_epochs=250 \
    --error_table='PerAtomMAE' \
    --swa \
    --start_swa=100 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --eval_interval=1 \
    --device=cuda \
    --restart_latest \
    --save_all_checkpoints \

date
