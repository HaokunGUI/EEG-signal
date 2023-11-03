torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name ssl \
    --model VQ_BERT \
    --learning_rate 5e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 400 \
    --mask_length 3 \
    --mask_ratio 0.15 \
    --codebook_num 4 \
    --codebook_item 1024 \
    --kernel_size 5 \
    --attn_head 8 \
    --e_layers 4 \
    --d_model 512 \
    --d_hidden 128 \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --dropout 0.3 \
    --min_space 1 \
    --mask_dropout 0.0 \
    --num_workers 8 \
    --enc_type "rel" \
    --mask_type "poisson"