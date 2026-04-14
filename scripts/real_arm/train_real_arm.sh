main_dir=RealArm

# ===== UPDATE THESE PATHS =====
DATA_PATH=/path/to/your/zarr_output

train_data_dir=$DATA_PATH/train.zarr/
eval_data_dir=$DATA_PATH/val.zarr/
train_instructions=instructions/real_arm/instructions.json
val_instructions=instructions/real_arm/instructions.json

# Dataset
dataset=RealArm
num_workers=4
B=16  # batch size, increase if you have more data/GPU memory
B_val=16
chunk_size=1
memory_limit=8

# Training
val_freq=2000
eval_only=false
lr=1e-4
backbone_lr=1e-6
lr_scheduler=constant
wd=1e-10
train_iters=100000  # adjust based on dataset size
use_compile=false
use_ema=false
lv2_batch_size=1

# Model
model_type=denoise3d
bimanual=false        # set true for dual arm
keypose_only=true     # set false if trajectory_length > 1
pre_tokenize=true
workspace_normalizer_buffer=0.05

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=120
num_attn_heads=8
num_vis_instr_attn_layers=2
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

ngpus=1

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --memory_limit $memory_limit \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --backbone_lr $backbone_lr \
    --lr_scheduler $lr_scheduler \
    --wd $wd \
    --train_iters $train_iters \
    --use_compile $use_compile \
    --use_ema $use_ema \
    --lv2_batch_size $lv2_batch_size \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
