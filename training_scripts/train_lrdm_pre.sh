# model configs
GRU_iters=5
optim_layer_input_clamp=1.0
depth_activation_format='exp'

# training configs
batch_size=1
lr=0.001
grad_loss_weight=10.0
intermediate_loss_weight=1.0
mask_out_rate=0.0

port=$(shuf -i 8001-40000 -n 1)

python main_2.py --dir_data /hy-tmp/dataset/stereomis_sample --data_name stereomis --split_json /hy-tmp/dataset/stereomis_sample/dataset_c3vd.json \
    --gpus 0,1 --multiprocessing --tcp_port $port  \
    --pretrain /hy-tmp/result/pretrain/OGNIDC_stereomis_250219_023126_without_DDI/model_best.pt \
    --lr $lr --batch_size $batch_size --milestones 18 24 28 32 36 --epochs 36 \
    --loss 1.0*SeqL1+1.0*SeqL2+$grad_loss_weight*SeqGradL1+1.0*Noise+0.001*SCC --intermediate_loss_weight $intermediate_loss_weight --training_depth_mask_out_rate $mask_out_rate \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --log_dir /hy-tmp/result/LRDM_test/OGNIDC_stereomis_loss_ognidc+lrdm_withfreeze \
    --num_diffusion_timesteps 1000 \
    --save "withpre"