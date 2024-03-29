torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name anomaly_detection \
    --model Ti_MAE \
    --learning_rate 1e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 20 \
    --e_layers 2 \
    --d_model 256 \
    --activation "gelu" \
    --train_batch_size 128 \
    --test_batch_size 128 \
    --dropout 0.3 \
    --linear_dropout 0.6 \
    --num_workers 10 \
    --use_scheduler \
    --weight_decay 1e-3 \
    --pretrained_path "/home/guihaokun/Time-Series-Pretrain/logging_ab/ssl/Ti_MAE/Ti_MAE_240125_0250/checkpoint/last.pth.tar" \
    --dataset TUAB \
    --root_path "/data/guihaokun/resample/tuh_eeg_abnormal/" \
    --log_dir "/home/guihaokun/Time-Series-Pretrain/logging_ab" 