CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 29501 train_joint_v2.py \
    --epochs 24 \
    --max_length 1280 \
    --log_interval 1000 \
    --sam_max_point_bs 4 \
    --learning_rate 0.001 \
    --data_path "./data/train_seg_all.jsonl" \
    --sam_checkpoint "./checkpoints/vit_b.pt" \
    --segment_llm_path "./checkpoints/sft_all_e1.pt" \
    --freeze_vision_projection False
