export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="/archive/vaibhav.agrawal/a-bev-of-the-latents/checkpoints"  # your save path
export CONFIG="./default_config.yaml"
export TRAIN_DATA="/archive/vaibhav.agrawal/a-bev-of-the-latents/datasetv9/rgb.jsonl" # your data jsonl file 
export LOG_PATH="$OUTPUT_DIR/log"

accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=512 \
    --spatial_column="oscr" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --lora_num 1 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --run_name="seethrough3d" \
    --debug=1 \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --stage1_epochs=1 \
    --stage2_steps=5000 \
    --checkpointing_steps=5000