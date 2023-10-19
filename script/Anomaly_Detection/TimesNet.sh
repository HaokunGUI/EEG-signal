torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name anomaly_detection \
    --model TimesNet \
    --normalize \
    --learning_rate 1e-3 \
    --patience 0 \
    --anomaly_ratio 0.083 \
    --d_hidden 8 \
    --num_kernels 5 \
    --d_model 16 \
    --e_layers 3 \
    --dropout 0.3 \
    --top_k 3 \
    --num_epochs 40