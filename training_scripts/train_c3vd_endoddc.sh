# model configs
GRU_iters=5
optim_layer_input_clamp=1.0
depth_activation_format='linear'

# training configs
batch_size=5
lr=0.0005
grad_loss_weight=10.0
intermediate_loss_weight=1.0
mask_out_rate=0.0

port=$(shuf -i 8001-40000 -n 1)


python main.py --dir_data /home/ren6/NAS_zerotier/yinheng/dataset/c3vd/c3vd_sample --data_name c3vd --split_json /home/ren6/NAS_zerotier/yinheng/dataset/c3vd/c3vd_sample/dataset_c3vd_sample.json \
    --gpus 1,2 --multiprocessing --tcp_port $port \
    --max_depth 100.0 \
    --lr $lr --batch_size $batch_size --milestones 18 24 28 32 36 --epochs 36 \
    --loss 1.0*SeqL1+1.0*SeqL2+$grad_loss_weight*SeqGradL1+10.0*Noise \
    --training_depth_mask_out_rate $mask_out_rate \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --log_dir /home/ren6/NAS_zerotier/yinheng/0215 \
    --num_diffusion_timesteps 1000 \
    --model_ch_mult 1 1 2 4 \
    --num_sampling_timesteps 20 \
    --save "ddc"