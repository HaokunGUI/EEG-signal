torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name ssl \
    --model SimMTM \
    --learning_rate 2e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 100 \
    --e_layers 2 \
    --d_model 256 \
    --activation "gelu" \
    --linear_dropout 0.5 \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --input_len 60 \
    --output_len 60 \
    --mask_ratio 0.15 \
    --positive_num 2 \
    --temperature 0.1 \
    --attn_head 8 \
    --dropout 0.3 \
    --num_workers 10 \
    --use_scheduler \
    --warmup_epochs 20