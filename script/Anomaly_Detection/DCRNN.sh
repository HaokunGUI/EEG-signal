torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name anomaly_detection \
    --model DCRNN \
    --use_curriculum_learning \
    --graph_type distance \
    --normalize \
    --use_fft \
    --use_pretrained \
    --pretrained_path "/home/guihaokun/Time-Series-Pretrain/logging/ssl/DCRNN_231017_185705/checkpoint/last.pth.tar"
