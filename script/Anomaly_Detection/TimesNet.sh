torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name anomaly_detection \
    --model TimesNet \
    --normalize \
    --learning_rate 5e-4 \
    --patience 0 \
    --anomaly_ratio 0.01 \
    --d_hidden 8 \
    --num_kernels 6 \
    --d_model 2 \
    --e_layers 2 \
    --dropout 0.3 \
    --top_k 3 \
    --num_epochs 40