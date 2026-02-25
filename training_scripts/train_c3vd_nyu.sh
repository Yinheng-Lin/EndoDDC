# model configs
GRU_iters=5
optim_layer_input_clamp=1.0
depth_activation_format='exp'

# training configs
batch_size=6
lr=0.001
grad_loss_weight=10.0
intermediate_loss_weight=1.0
mask_out_rate=0.0

port=$(shuf -i 8001-40000 -n 1)

python main.py --dir_data /mnt/data2_hdd/yinheng/dataset/c3vd/c3vd_h5 --data_name c3vd --split_json /mnt/data2_hdd/yinheng/dataset/c3vd/data_c3vd.json \
    --gpus 1 --tcp_port $port  \
    --pretrain /mnt/data2_hdd/yinheng/OGNI-DC-main/src/checkpoints/NYU_best_performance.pt \
    --lr $lr --batch_size $batch_size --milestones 18 24 28 32 36 --epochs 36 \
    --loss 1.0*SeqL1+1.0*SeqL2+$grad_loss_weight*SeqGradL1 --intermediate_loss_weight $intermediate_loss_weight --training_depth_mask_out_rate $mask_out_rate \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --log_dir /mnt/data2_hdd/yinheng/OGNI-DC-main/result/c3vd_nyu \
    --save "train_c3vd_nyu"