# model configs
GRU_iters=1
optim_layer_input_clamp=1.0
depth_activation_format='exp'

# training configs
batch_size=6
lr=0.001
grad_loss_weight=10.0
intermediate_loss_weight=1.0
mask_out_rate=0.0

port=$(shuf -i 8001-40000 -n 1)

python main.py --dir_data /hy-tmp/c3vd_test --data_name c3vd --split_json /hy-tmp/c3vd_test/c3vd_test.json \
    --gpus 0,1 --multiprocessing --tcp_port $port  \
    --max_depth 100.0 \
    --lr $lr --batch_size $batch_size --milestones 18 24 28 32 36 --epochs 36 \
    --loss 1.0*SeqL1+1.0*SeqL2+$grad_loss_weight*SeqGradL1 --intermediate_loss_weight $intermediate_loss_weight --training_depth_mask_out_rate $mask_out_rate \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --log_dir /hy-tmp/result/test_image/ \
    --save "ognidc"