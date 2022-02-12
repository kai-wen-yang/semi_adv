#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J baselineseed1   # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=2     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10       # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:2            # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p short4,p40,gpu4             # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 4-24:00:00            # 运行的最长时间 day-hour:minute:second
#SBATCH -o results/parallel.out       # 打印输出的文件

conda activate cmc
#python -m torch.distributed.launch --nproc_per_node 2 fixmatch_stl_perp.py --seed 1 --dataset stl10 --batch-size 32 --expand-labels --out ./results/stl10_s1_e0.1_a1.0_g1_perp --dim 512 --eps 0.1 --alpha 1.0 --gamma 1;
python -m torch.distributed.launch --nproc_per_node 2 fixmatch_stl.py --seed 1 --dataset stl10 --batch-size 32 --expand-labels --out ./results/stl10_s1_e0.1_a1.0_g1_adv3 --dim 512 --vae_path ../FixMatch-pytorch/results/vae_dim512_kl0.1_stl/model_epoch222.pth --eps 0.1 --alpha 1.0 --gamma 1;

#python -m torch.distributed.launch --nproc_per_node 4 fixmatch_stl.py --seed 1 --dataset stl10 --batch-size 16 --expand-labels --out ./results/stl10_s1_e0.1_a1.0_g1_adv3 --dim 512 --vae_path ../FixMatch-pytorch/results/vae_dim512_kl0.1_stl/model_epoch222.pth --eps 0.1 --alpha 1.0 --gamma 1;

#python fixmatch_adv3.py --seed 1 --dataset cifar10 --num-labeled 250 --expand-labels --out ./results/cifar10_l250_s1_e0.0_a1.0_g1_adv3 --dim 512 --vae_path ../CLAE_VAE/results/vae_dim512_kl0.1/model_epoch222.pth --eps 0.0 --alpha 1.0 --gamma 1;

