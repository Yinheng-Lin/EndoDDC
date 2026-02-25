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

python main.py --dir_data /home/ren6/NAS/yinheng/dataset/StereoMIS/StereoMIS_h5 --data_name stereomis --split_json /home/ren6/NAS/yinheng/dataset/StereoMIS/StereoMIS.json \
    --gpus 2 --tcp_port $port  \
    --pretrain /mnt/data2_hdd/yinheng/OGNI-DC-main/src/checkpoints/NYU_best_performance.pt \
    --lr $lr --batch_size $batch_size --milestones 18 24 28 32 36 --epochs 36 \
    --loss 1.0*SeqL1+1.0*SeqL2+$grad_loss_weight*SeqGradL1 --intermediate_loss_weight $intermediate_loss_weight --training_depth_mask_out_rate $mask_out_rate \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --log_dir /home/ren6/NAS/yinheng/dataset/OGNIDC_RESULT_ \
    --save "train_StereoMIS_nyu"