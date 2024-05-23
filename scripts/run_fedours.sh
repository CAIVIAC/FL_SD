#!/bin/bash
# bash ./scripts/run_fedours.sh

# ## baselines --------------------------------------------------------------------------------------
# ## one user for evaluating KD (for main_bkd_sng.py)
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 12 --results_save full
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  7 --results_save full

# ## variants --------------------------------------------------------------------------------------
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  2 --results_save rlt2
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  3 --results_save rlt3
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  4 --results_save rlt4
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  5 --results_save rlt5
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  6 --results_save rlt6
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  7 --results_save rlt7
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  8 --results_save rlt8
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  9 --results_save rlt9
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 10 --results_save rlt10
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 11 --results_save rlt11

# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  2 --results_save rlt2
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  3 --results_save rlt3
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  4 --results_save rlt4
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  5 --results_save rlt5
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  6 --results_save rlt6

 
 
## singleKD variants --------------------------------------------------------------------------------------
## full means complete network; rltK means network using K-th output
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  2 --results_save rlt2
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  3 --results_save rlt3
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  4 --results_save rlt4
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  5 --results_save rlt5
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  6 --results_save rlt6
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  7 --results_save rlt7
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  8 --results_save rlt8
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  9 --results_save rlt9
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 10 --results_save rlt10
# python main_singleKD.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 11 --results_save rlt11

# python main_singleKD.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  2 --results_save rlt2
# python main_singleKD.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  3 --results_save rlt3
# python main_singleKD.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  4 --results_save rlt4
# python main_singleKD.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  5 --results_save rlt5
# python main_singleKD.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  6 --results_save rlt6




# #### SKD2C: Synchronized KD, FL learned teacher and FL learned student ----------------------------------
# ## more clients on mobilenet
# python main_skd2c.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_skd2c.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_skd2c.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# ## few clinets on mobilenet
# python main_skd2c.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_skd2c.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3

# ## few clinets on resnet18
# python main_skd2c.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_skd2c.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# ## more clients on resnet18
# python main_skd2c.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_skd2c.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_skd2c.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5



# ## SNG2H variants --------------------------------------------------------------------------------------
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  2 --results_save rlt2
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  3 --results_save rlt3
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  4 --results_save rlt4
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  5 --results_save rlt5
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  6 --results_save rlt6
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  7 --results_save rlt7
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  8 --results_save rlt8
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  9 --results_save rlt9
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 10 --results_save rlt10
# python main_single2h.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 11 --results_save rlt11







# Table 3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final






# ### finetuning 5 epochs ----------------------- GPU0
# # Table 4     s 100   body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body

# # Table 4    s 100   head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head

# # Table 4    s 100   full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full







# # Table 16 shard_per_user  100   GPU0
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full






# # Table ablation shard_per_user  100   GPU0
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# # Table ablation shard_per_user  50   GPU0
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# # Table ablation shard_per_user  10    GPU0
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1 --KD_mode 0 --ol  3 --results_save abla1
# # Table ablation shard_per_user  100   GPU0
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001
# # Table ablation shard_per_user  50   GPU0
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001
# # Table ablation shard_per_user  10    GPU0
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1001 --KD_mode 0 --ol  3 --results_save abla1001









# ############################################
# # resnet18
# ############################################

# # # Table 16 shard_per_user  100   GPU0
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full




# ############################################
# # resnet50
# ############################################

# # # Table 16 shard_per_user  100   GPU0
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full





# ############################################
# # 4covNet
# ############################################

# Table 12 shard_per_user  10   GPU0
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs  80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs  32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

### finetuning 5 epochs ----------------------- GPU0
# Table 13    s 10   full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 10 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 0 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full