torchrun \
    --standalone \
    --nproc_per_node=2 \
    run.py \
    --task_name ssl \
    --model BERT \
    --learning_rate 2e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 150 \
    --codebook_item 1024 \
    --e_layers 2 \
    --d_model 256 \
    --hidden_channels 16 \
    --activation "gelu" \
    --train_batch_size 128 \
    --test_batch_size 128 \
    --dropout 0.7 \
    --num_workers 8 \
    --mask_type "poisson" \
    --use_scheduler \
    --warmup_epochs 20