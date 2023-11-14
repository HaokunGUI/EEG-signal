torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name anomaly_detection \
    --model DCRNN \
    --graph_type distance \
    --normalize \
    --use_fft \
    --learning_rate 1e-4 \
    --data_augment \
    --patience 0 \
    --pretrained_path "/home/guihaokun/Time-Series-Pretrain/logging/anomaly_detection/DCRNN/DCRNN_231024_160312/checkpoint/last.pth.tar"\
    --dropout 0.5 \
    --balanced \
    --num_epochs 60