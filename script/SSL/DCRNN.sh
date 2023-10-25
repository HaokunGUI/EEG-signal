torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name ssl \
    --model DCRNN \
    --use_curriculum_learning \
    --graph_type distance \
    --learning_rate 1e-3 \
    --normalize \
    --use_fft \
    --data_augment \
    --num_epochs 120 \
    --dropout 0.5 \
    --patience 0

