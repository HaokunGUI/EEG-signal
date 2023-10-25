torchrun \
    --standalone \
    --nproc_per_node=1 \
    run.py \
    --task_name anomaly_detection \
    --model DCRNN \
    --graph_type distance \
    --normalize \
    --use_fft \
    --learning_rate 1e-3 \
    --data_augment \
    --use_pretrained \
    --patience 0 \
    --pretrained_path "/home/guihaokun/Time-Series-Pretrain/logging/ssl/DCRNN/DCRNN_231024_002838/checkpoint/last.pth.tar"\
    --dropout 0.5 \
    --balanced \
    --num_epochs 60