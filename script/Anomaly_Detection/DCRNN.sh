torchrun \
    --standalone \
    --nproc_per_node=1 \
    run.py \
    --task_name anomaly_detection \
    --model DCRNN \
    --graph_type distance \
    --normalize \
    --use_fft \
    --learning_rate 1e-4 \
    --data_augment \
    --balanced \ 
    --use_pretrained \
    --pretrained_path "/home/guihaokun/Time-Series-Pretrain/logging/ssl/DCRNN_231017_185705/checkpoint/last.pth.tar"
