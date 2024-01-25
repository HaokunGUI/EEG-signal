torchrun \
    --standalone \
    --nproc_per_node=3 \
    run.py \
    --task_name anomaly_detection \
    --model BERT \
    --learning_rate 1e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 60 \
    --codebook_item 1024 \
    --e_layers 2 \
    --d_model 256 \
    --hidden_channels 16 \
    --activation "gelu" \
    --linear_dropout 0.7 \
    --train_batch_size 128 \
    --test_batch_size 128 \
    --dropout 0.3 \
    --num_workers 10 \
    --use_scheduler \
    --weight_decay 1e-4 \
    --dataset TUAB \
    --root_path "/data/guihaokun/resample/tuh_eeg_abnormal/" \
    --log_dir "/home/guihaokun/Time-Series-Pretrain/logging_ab" \
    --pretrained_path "/home/guihaokun/Time-Series-Pretrain/logging_ab/ssl/BERT/BERT_240124_194027/checkpoint/last.pth.tar"