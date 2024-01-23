torchrun \
    --standalone \
    --nproc_per_node=2 \
    run.py \
    --task_name anomaly_detection \
    --model Ti_MAE \
    --learning_rate 1e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 60 \
    --e_layers 2 \
    --d_model 256 \
    --activation "gelu" \
    --train_batch_size 128 \
    --test_batch_size 128 \
    --dropout 0.3 \
    --linear_dropout 0.6 \
    --num_workers 12 \
    --use_scheduler \
    --balanced \
    --weight_decay 1e-4 \
    --pretrained_path "/home/guihaokun/Time-Series-Pretrain/logging/ssl/Ti_MAE/Ti_MAE_240106_021019/checkpoint/last.pth.tar" \
    --input_len 12