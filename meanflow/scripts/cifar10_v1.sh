CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --master_port=12347 \
    train.py \
    --output_dir=./logs/2025.08.06KST20:20 \
    --dataset=cifar10 \
    --batch_size=384 \
    --lr=0.0006 \
    --eval_frequency=50 \
    --epochs=16000 \
    --compute_fid \
    --log_per_step=100 \
    --tr_sampler=v1 \
    --P_mean_t -0.6 \
    --P_std_t 1.6 \
    --P_mean_r -4.0 \
    --P_std_r 1.6 \
    --warmup_epochs 200 \
    --norm_p 0.75 \
    --ratio 0.75 \
    --dropout 0.2 \
    --use_edm_aug
