#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J baselineseed1   # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=4     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10       # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:4            # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p p40          # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 4-24:00:00            # 运行的最长时间 day-hour:minute:second
#SBATCH -o results/random2.out       # 打印输出的文件

conda activate cmc
python -m torch.distributed.launch --nproc_per_node 4 fixmatch_random_init_custom_minus.py  --seed 1 --dataset cifar100 --num-labeled 2500 --expand-labels \
--amp --opt_level O2 --wdecay 0.001 --out ./results/cifar100_randominit_minus_start0.1_step0_epsmax0.1 --batch-size 16 --start 0.1 --step 0 --eps_max 0.1
