#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J vaedim256_kl0   # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10       # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1            # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p  "short,p40,gpu,short4,gpu4"               # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 0-24:00:00            # 运行的最长时间 day-hour:minute:second
#SBATCH -o results/adv_cifar5.out       # 打印输出的文件

conda activate cmc
#sleep 5h
python vae_pretrain_stl.py --dim 512 --re 1 1 1 --kl 0.1 --batch_size 512 --save_dir ./results/vae_dim512_kl0.1_stl --dataset stl10 --lr 1e-4
#python vae_latent_pretrain2.py --dim 128 --re 10 --ce 1  --save_dir ./results/latent_vae_dim128_re10_ce1 --dataset cifar10;




