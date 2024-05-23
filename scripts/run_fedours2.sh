#!/bin/bash#!/bin/bash
# bash ./scripts/run_fedours2.sh

## variants ---------------------------------------------------------------------------------------
## full means complete network; rltK means network using K-th output
## for evaluating KD (for main_bkd_babu.py)
## strong teachers --------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full
## weak teachers ----------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full

# ## strong students --------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# ## weak students ----------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5

# ## strong students --------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# ## weak students ----------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5

# ## strong students --------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save rlt7
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save rlt7
# ## weak students ----------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save rlt7
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save rlt7
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  7 --results_save rlt7

# ## strong students --------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  9 --results_save rlt9
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  9 --results_save rlt9
# ## weak students ----------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  9 --results_save rlt9
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  9 --results_save rlt9
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  9 --results_save rlt9





# Table 3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final






# ### finetuning 5 epochs ----------------------- GPU2
# # Table 4     s 10   body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body

# # Table 4    s 10   head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head

# # Table 4    s 10   full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full





# # Table 16 shard_per_user  10    GPU2
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full







# # Table ablation shard_per_user  100   GPU2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# # Table ablation shard_per_user  50   GPU2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# # Table ablation shard_per_user  10    GPU2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 3 --KD_mode 0 --ol  3 --results_save abla3
# # Table ablation shard_per_user  100   GPU2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003
# # Table ablation shard_per_user  50   GPU2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003
# # Table ablation shard_per_user  10    GPU2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1003 --KD_mode 0 --ol  3 --results_save abla1003






# ############################################
# # resnet18
# ############################################

# # # Table 16 shard_per_user  10    GPU2
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full



# ############################################
# # resnet50
# ############################################

# # Table 16 shard_per_user  10    GPU2
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full





# ############################################
# # 4covNet
# ############################################

# Table 12 shard_per_user  2    GPU2
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  2 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  2 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  2 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  2 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  2 --epochs  80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  2 --epochs  32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

### finetuning 5 epochs
# Table 13    s 2   full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 2 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 2 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 2 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 2 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 2 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 2 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full