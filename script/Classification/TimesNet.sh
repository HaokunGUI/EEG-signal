torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name classification \
    --model TimesNet \
    --learning_rate 5e-4 \
    --patience 0 \
    --d_hidden 8 \
    --num_kernels 6 \
    --d_model 16 \
    --e_layers 3 \
    --dropout 0.2 \
    --top_k 3 \
    --num_epochs 60 \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --use_scheduler \
    --num_workers 8 \
    --normalize