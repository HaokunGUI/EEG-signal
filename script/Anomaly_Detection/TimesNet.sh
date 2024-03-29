torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name anomaly_detection \
    --model TimesNet \
    --learning_rate 1e-3 \
    --patience 0 \
    --d_hidden 8 \
    --num_kernels 6 \
    --d_model 16 \
    --e_layers 3 \
    --dropout 0.3 \
    --top_k 3 \
    --num_epochs 20 \
    --train_batch_size 128 \
    --test_batch_size 128 \
    --use_scheduler \
    --num_workers 10 \
    --dataset TUAB \
    --root_path "/data/guihaokun/resample/tuh_eeg_abnormal/" \
    --log_dir "/home/guihaokun/Time-Series-Pretrain/logging_ab" \
    --normalize