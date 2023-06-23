#!/bin/bash
conda activate guided_diffusion
python -m scripts.pl_train --log_dir /store/CIA/js2164/models/gd_cifar10 \
                            --log_name prior --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 32 --learn_sigma True \
                            --dropout 0.3 --noise_schedule cosine --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False \
                            --use_scale_shift_norm True --batch_size 128 --lr=1e-4 --lr_anneal_steps 5000 --data_dir /store/CIA/js2164/data \
                            --dataset cifar10 --accumulate_grad_batches 1 --grad_clip 1