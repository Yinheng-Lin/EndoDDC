GRU_iters=5
test_augment=0
optim_layer_input_clamp=1.0
depth_activation_format='exp'

ckpt=

python main.py --dir_data /home/ren6/NAS/yinheng/dataset/StereoMIS/StereoMIS_h5 --data_name stereomis --split_json /home/ren6/NAS/yinheng/dataset/StereoMIS/StereoMIS.json \
    --gpus 2 --max_depth 250.0 --num_sample 500 \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --test_only --test_augment $test_augment --pretrain $ckpt \
    --log_dir /home/ren6/NAS/yinheng/dataset/OGNIDC_result_test_nyu \
    --save "test_stereomis" \
    --save_result_only