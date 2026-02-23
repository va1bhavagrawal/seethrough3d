#!/bin/bash
#SBATCH --job-name=vaibhav
#SBATCH --output=%j.out
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=4           
#SBATCH --mem=150G
#SBATCH --gres=gpu:4                    
#SBATCH --partition=ada

# chetna
# export MODEL_DIR="black-forest-labs/FLUX.1-Kontext-dev" # your flux path
export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="/archive/vaibhav.agrawal/a-bev-of-the-latents/easycontrol_cuboids"  # your save path
export CONFIG="./default_config.yaml"
export TRAIN_DATA="/archive/vaibhav.agrawal/a-bev-of-the-latents/datasetv7_superhard/cuboids__upto_4subjects.jsonl" # your data jsonl file 
export LOG_PATH="$OUTPUT_DIR/log"
export INFERENCE_EMBEDS_DIR="/archive/vaibhav.agrawal/a-bev-of-the-latents/inference_embeds_datasetv7_superhard"

export WANDB_API_KEY=f27c837d8d7d0c8d79f3eb1de21fa78233c03be6

# kotak
# export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
# export OUTPUT_DIR="/archive/vaibhav.agrawal/a-bev-of-the-latents/easycontrol_cuboids"  # your save path
# export CONFIG="./default_config.yaml"
# export TRAIN_DATA="/archive/vaibhav.agrawal/a-bev-of-the-latents/datasetv6/cuboids.jsonl" # your data jsonl file
# export LOG_PATH="$OUTPUT_DIR/log"
# export INFERENCE_EMBEDS_DIR="/archive/vaibhav.agrawal/a-bev-of-the-latents/inference_embeds_flux2"

# kotak
# export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
# export OUTPUT_DIR="./easycontrol_cuboids"  # your save path
# export CONFIG="./default_config.yaml"
# export TRAIN_DATA="/home/venky/vaibhav.agrawal/a-bev-of-the-latents/datasets/actual_data/datasetv6/cuboids.jsonl" # your data jsonl file
# export LOG_PATH="$OUTPUT_DIR/log"
# export INFERENCE_EMBEDS_DIR="/home/venky/vaibhav.agrawal/a-bev-of-the-latents/caching/inference_embeds_flux2"

# i love this.
accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=512 \
    --subject_column="None" \
    --spatial_column="cv" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --lora_num 1 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --run_name="rgb__r1" \
    --debug=1 \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --inference_embeds_dir $INFERENCE_EMBEDS_DIR \
    --validation_prompt "a photo of sedan and pickup truck and suv amongst autumn-colored trees along a winding river" "a photo of cow and suv on a sandy beach with palm trees swaying in the breeze" "a photo of table and horse and suv in a dense pine forest with tall trees reaching the sky" \
    --num_train_epochs=1 \
    --validation_steps=5000000000000 \
    --checkpointing_steps=2500 \
    --spatial_test_images "cuboids/sedan__pickup_truck__suv/005/cuboids.png" "cuboids/cow__suv/008/cuboids.png" "cuboids/table__horse__suv/007/cuboids.png" \
    --subject_test_images None \
    --test_h 512 \
    --test_w 512 \
    --num_validation_images=1

    # --run_name="semantic_info_from_cuboid_cond" \
    # --run_name="datasetv8__0.8_0.1_0.1" \
    # --pretrained_lora_path="/archive/vaibhav.agrawal/a-bev-of-the-latents/easycontrol_cuboids/wireframe/epoch-0__checkpoint-5000/lora.safetensors" \
    # --pretrained_lora_path="/archive/vaibhav.agrawal/a-bev-of-the-latents/easycontrol_cuboids/rgb/epoch-0__checkpoint-7500/lora.safetensors" \
    # --pretrained_lora_path="/archive/vaibhav.agrawal/a-bev-of-the-latents/easycontrol_cuboids/datasetv9__wireframe_best_case/epoch-0__checkpoint-3888/lora.safetensors" \