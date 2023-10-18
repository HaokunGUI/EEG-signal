torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name ssl \
    --model DCRNN \
    --use_curriculum_learning \
    --graph_type distance \
    --learning_rate 5e-4 \
    --normalize \
    --use_fft \
    --data_augment
