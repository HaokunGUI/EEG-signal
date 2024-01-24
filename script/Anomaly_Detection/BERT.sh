export OMP_NUM_THREADS=4 
torchrun \
    --standalone \
    --nproc_per_node=2 \
    --rdzv_endpoint localhost:6000  \
    --rdzv_backend c10d \
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
    --linear_dropout 0.6 \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --dropout 0.3 \
    --num_workers 8 \
    --use_scheduler \
    --balanced \
    --weight_decay 1e-4 \
    --pretrained_path "/home/guihaokun/Time-Series-Pretrain/logging/ssl/BERT/BERT_240124_012022/checkpoint/last.pth.tar"\
    --input_len 12