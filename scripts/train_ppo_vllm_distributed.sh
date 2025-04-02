#!/usr/bin/env bash

# Basic configuration
DATA=mmmath
MODEL=2.5it

PROJECT=MAYE-$DATA
TASK_NAME=MAYE-${MODEL}-${DATA}-5e6-up32-gbz896

# Dataset selection based on DATA
if [[ "$DATA" == "geo3k" ]]; then
    DATADIR=geometry3k
    TRAIN_DATASET_NAME=geometry3k_rl_v0
    VALIDATION_DATASET_NAME=geometry3k_validation_v0
    TEST_DATASET_NAME=geometry3k_test_v0

elif [[ "$DATA" == "mmmath" ]]; then
    DATADIR=mm_math
    TRAIN_DATASET_NAME=mm_math_rl_v0
    VALIDATION_DATASET_NAME=mm_math_validation_v0
    TEST_DATASET_NAME=math_verse_test_v0

else
    echo "Error: Invalid data value '$DATA'. Use 'geo3k', 'mmmath'."
    exit 1
fi

# Model selection based on MODEL
if [[ "$MODEL" == "2.5it" ]]; then
    MODEL_NAME="Qwen2.5-VL-7B-Instruct"
    PROCESSOR_NAME="Qwen2.5-VL-7B-Instruct"
    MODEL_CLASS_NAME="Qwen2_5_VLForConditionalGeneration"
elif [[ "$MODEL" == "2it" ]]; then
    MODEL_NAME="Qwen2-VL-7B-Instruct"
    PROCESSOR_NAME="Qwen2-VL-7B-Instruct"
    MODEL_CLASS_NAME="Qwen2VLForConditionalGeneration"
else
    echo "Error: Invalid tag value '$MODEL'. Use '2.5it' or '2it'."
    exit 1
fi

# File paths
MODEL_PATH=/tmp/ckpts/$MODEL_NAME
PROCESSOR_PATH=/tmp/ckpts/$PROCESSOR_NAME
OUTPUT_DIR=/tmp/ckpts/$MODEL_NAME-$DATADIR_NAME
TRAIN_DATASET_PATH=/tmp/MAYE/datasets/$DATADIR/${TRAIN_DATASET_NAME}.jsonl
VALIDATION_DATASET_PATH=/tmp/MAYE/datasets/$DATADIR/${VALIDATION_DATASET_NAME}.jsonl
TEST_DATASET_PATH=/tmp/MAYE/datasets/$DATADIR/${TEST_DATASET_NAME}.jsonl

# Hyper-Parameters
BATCH_SIZE=128
CLIP_GRAD_NORM=1.0
ENABLE_ACTIVATION_CHECKPOINTING=True
ENABLE_ACTIVATION_OFFLOADING=False
ENABLE_LR_SCHEDULER=True
EPSILON_HIGH=0.2
EPSILON_LOW=0.2
FORWARD_BATCH_SIZE=16
GAMMA=1.0
GRADIENT_ACCUMULATION_STEPS=2
KL_LOSS_COEFF=0.001
KL_REWARD_COEFF=0.000
LOG_EVERY_N_STEPS=1
LR=5e-6
MAX_TOKENS=2048
MIN_RESPONSE_LENGTH=1
NUM_EPOCHS=30
PENALISE_NO_EOS=True
PPO_BATCH_SIZE=4
PPO_EPOCH=1
REWARD_PENALTY=-0.1
SAVE_EVAL_FILES=False
SAVE_EVERY_N_EPOCHS=0
SEED=0
TEMPERATURE=1.0
TRAIN_CONNECTOR=False
TRAIN_LLM=True
TRAIN_VIT=False
USE_CHAT_TEMPLATE=True
VLLM_GPU_MEMORY_UTILIZATION=0.75
WHITEN_REWARDS=True

export VLLM_LOGGING_LEVEL=WARNING

DEVICES="0,1,2,3,4,5,6,7"

NUM_DEVICES=$(echo $DEVICES | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
    --nproc_per_node="$((NUM_DEVICES - 1))" \
    --master_port=12345 \
    recipes/full_ppo_vllm_distributed.py \
    --config-name="full_ppo_vllm_distributed" \
    config.pretrained_model_name_or_path="${MODEL_PATH}" \
    model._target_=transformers."${MODEL_CLASS_NAME}" \
    processor._target_=transformers.AutoProcessor.from_pretrained \
    processor.pretrained_model_name_or_path="${PROCESSOR_PATH}" \
    dataset._target_=maye.datasets.MathGenerationDataset \
    validation_dataset._target_=maye.datasets.MathGenerationDataset \
    test_dataset._target_=maye.datasets.MathGenerationDataset \
    collate_fn._target_=maye.utils.collate.collate_rlhf_vllm \
    validation_collate_fn._target_=maye.utils.collate.collate_generation_vllm \
    test_collate_fn._target_=maye.utils.collate.collate_generation_vllm \
    loss._target_=maye.rlhf.loss.PPOLoss \
    loss.epsilon_low="${EPSILON_LOW}" \
    loss.epsilon_high="${EPSILON_HIGH}" \
    loss.kl_loss_coeff="${KL_LOSS_COEFF}" \
    vllm.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
    dataset.dataset_path="${TRAIN_DATASET_PATH}" \
    validation_dataset.dataset_path="${VALIDATION_DATASET_PATH}" \
    test_dataset.dataset_path="${TEST_DATASET_PATH}" \
    dataset.use_chat_template="${USE_CHAT_TEMPLATE}" \
    generation_kwargs.temperature="${TEMPERATURE}" \
    generation_kwargs.max_tokens="${MAX_TOKENS}" \
    optimizer.lr="${LR}" \
    compile=True \
    output_dir="${OUTPUT_DIR}" \
    train_vit="${TRAIN_VIT}" \
    train_connector="${TRAIN_CONNECTOR}" \
    train_llm="${TRAIN_LLM}" \
    batch_size="${BATCH_SIZE}" \
    forward_batch_size="${FORWARD_BATCH_SIZE}" \
    ppo_epochs="${PPO_EPOCH}" \
    ppo_batch_size="${PPO_BATCH_SIZE}" \
    num_epochs="${NUM_EPOCHS}" \
    gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    penalise_no_eos="${PENALISE_NO_EOS}" \
    reward_penalty="${REWARD_PENALTY}" \
    min_response_length="${MIN_RESPONSE_LENGTH}" \
    seed="${SEED}" \
    clip_grad_norm="${CLIP_GRAD_NORM}" \
    enable_activation_checkpointing="${ENABLE_ACTIVATION_CHECKPOINTING}" \
    enable_activation_offloading="${ENABLE_ACTIVATION_OFFLOADING}" \
    kl_reward_coeff="${KL_REWARD_COEFF}" \
    whiten_rewards="${WHITEN_REWARDS}" \
    gamma="${GAMMA}" \
    log_every_n_steps="${LOG_EVERY_N_STEPS}" \
    save_every_n_epochs="${SAVE_EVERY_N_EPOCHS}" \
    save_eval_files="${SAVE_EVAL_FILES}" \
    enable_lr_scheduler="${ENABLE_LR_SCHEDULER}" \
    metric_logger.project="${PROJECT}" \
    metric_logger.name="${TASK_NAME}" \
    metric_logger.tags=["${TASK_NAME}"] \
