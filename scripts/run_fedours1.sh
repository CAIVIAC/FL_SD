#!/bin/bash
# bash ./scripts/run_fedours1.sh

## BKD: baseline KD, fix teacher and FL learned student -------------------------------------------
## few clinets on mobilenet
# python main_bkd_sng.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_bkd_sng.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
## more clients on mobilenet
# python main_bkd_sng.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_bkd_sng.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_bkd_sng.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
## few clinets on resnet18
# python main_bkd_sng.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_bkd_sng.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
## more clients on resnet18
# python main_bkd_sng.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_bkd_sng.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_bkd_sng.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5


# #### EKD: equidistant KD, FL learned teacher and FL learned student ----------------------------------
# ## more clients on mobilenet
# python main_ekd.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_ekd.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_ekd.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3

# ## few clinets on mobilenet
# python main_ekd.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_ekd.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3


# ## few clinets on resnet18
# python main_ekd.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_ekd.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# ## more clients on resnet18
# python main_ekd.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_ekd.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_ekd.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5





# Table 3
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final





# ### finetuning 5 epochs ----------------------- CPU1
# # Table 4     s 50   body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_body

# # Table 4    s 50   head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part head --aggr_part head --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_head

# # Table 4    s 50   full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full





# # Table 16 shard_per_user  50   GPU1
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full







# # Table ablation shard_per_user  100   GPU1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# # Table ablation shard_per_user  50   GPU1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# # Table ablation shard_per_user  10    GPU1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 2 --KD_mode 0 --ol  3 --results_save abla2
# # Table ablation shard_per_user  100   GPU1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002
# # Table ablation shard_per_user  50   GPU1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002
# # Table ablation shard_per_user  10    GPU1
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002
# python main_ours.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user  10 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1002 --KD_mode 0 --ol  3 --results_save abla1002






# ############################################
# # resnet18
# ############################################

# # # Table 16 shard_per_user  50   GPU1
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet18   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full



# ############################################
# # resnet50
# ############################################

# # # Table 16 shard_per_user  50   GPU1
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user  50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user  50 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
# python main_ours.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user  50 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 50 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
# python main_ours_ft.py --dataset cifar100 --model resnet50   --num_classes 100 --shard_per_user 50 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full





# ############################################
# # 4covNet
# ############################################

# Table 12 shard_per_user  5   GPU1
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  5 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  5 --epochs  80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  5 --epochs  32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  5 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  5 --epochs  80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final
python main_ours.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user  5 --epochs  32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final

### finetuning 5 epochs
# Table 13    s 5   full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 5 --epochs 320 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 5 --epochs  80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 5 --epochs  32 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 5 --epochs 320 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  1 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 5 --epochs  80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep  4 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full
python main_ours_ft.py --dataset cifar10 --model cnn   --num_classes 10 --shard_per_user 5 --epochs  32 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --local_upt_part full --aggr_part full --momentum 0.90 --wd 0.0 --seed 1 --gpu 1 --temperature 1 --KD_weight 1005 --KD_mode 0 --ol  3 --results_save final_ft_full