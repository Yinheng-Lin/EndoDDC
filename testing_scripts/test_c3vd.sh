GRU_iters=5
test_augment=0
optim_layer_input_clamp=1.0
depth_activation_format='exp'

ckpt=

python main.py --dir_data /mnt/data2_hdd/yinheng/dataset/c3vd_h5 --data_name c3vd --split_json /mnt/data2_hdd/yinheng/dataset/c3vd/data_c3vd.json \
    --gpus 1 --max_depth 100.0 --num_sample 500 \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --test_only --test_augment $test_augment --pretrain $ckpt \
    --log_dir /mnt/data2_hdd/yinheng/OGNI-DC-main/result/inference \
    --save "test_c3vd" \
    --save_result_only
