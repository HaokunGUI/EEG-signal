torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name ssl \
    --model BERT \
    --learning_rate 2e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 150 \
    --e_layers 2 \
    --d_layers 1 \
    --d_model 256 \
    --mask_ratio 0.75 \
    --activation "gelu" \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --dropout 0.3 \
    --linear_dropout 0.5 \
    --num_workers 8 \
    --mask_type "poisson" \
    --use_scheduler \
    --warmup_epochs 20 