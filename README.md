# <img src="./assets/logo.png" alt="Logo" width="28" style="vertical-align: middle;"/>Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme

<div align="center">
<p align="center">
  📖 <a href="./assets/paper.pdf">Paper</a> |
  🤗 <a href="https://huggingface.co/datasets/ManTle/MAYE">Datasets</a>
</p>
</div>

## 📚 Overview

This project presents **MAYE**, a transparent and reproducible framework and a comprehensive evaluation scheme for applying reinforcement learning (RL) to vision-language models (VLMs). The codebase is built entirely from scratch without relying on existing RL toolkits.

Key contributions include:

- **🧱 From-scratch RL framework**: A minimal training pipeline using standard libraries (Transformers, FSDP2, vLLM), enabling full visibility and extensibility in VLM RL training.
- **📊 Standardized evaluation scheme**: A unified protocol for measuring training dynamics and reflective behavior, filling a critical gap in RL evaluation for VLMs.
- **🔍 Empirical insights**: Experiments across datasets and models uncover trends in response length, reflection patterns, and demonstrate that RL consistently outperforms supervised fine-tuning (SFT) in generalization—even with high-quality data.

Together, the framework and evaluation scheme aim to establish a reproducible baseline and encourage broader adoption of RL in multimodal reasoning research.

**For more details on the training framework, evaluation scheme, and experimental analysis, please refer to our paper.**

## 🧠 Design Philosophy

This project is not intended to compete with existing high-performance RL libraries such as [TRL](https://github.com/huggingface/trl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [verl](https://github.com/OpenRLHF/OpenRLHF), or [AReaL](https://github.com/inclusionAI/AReaL).  
Instead, it aims to offer a transparent, lightweight, and educational framework that exposes the core logic of RL training for VLMs—without heavy abstraction or engineering overhead.  
In spirit, it is similar to [OpenAI SpinningUp](https://spinningup.openai.com/en/latest/): not focused on performance or feature completeness, but on being a clear and reproducible entry point for understanding and building VLM-RL systems.


The code is validated on 8 GPUs using a hybrid setup:

- Ranks 0–6: handle distributed training via [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md).
- Rank 7: is dedicated to high-throughput inference using [vLLM](https://github.com/vllm-project/vllm).

This separation ensures smooth integration of training and generation within a unified pipeline, and allows researchers to easily trace, debug, and extend every component.

> 💡 For debugging under distributed execution, we recommend using [`torch.distributed.breakpoint()`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.breakpoint) to enter an interactive shell on all ranks.

🙏 **Acknowledgements**:  
The training logic is heavily inspired by [torchtune](https://github.com/pytorch/torchtune), a well-designed and native PyTorch post-training library with clean abstractions.  
The use of vLLM for response generation—separated onto a dedicated inference rank—follows the early design in [TRL](https://github.com/huggingface/trl), which adopted this pattern before supporting tensor parallel inference. (As of now, TRL's latest version has integrated TP-compatible vLLM inference.)

### 🍕 Preliminary

While (GRPO) has become the most widely used RL algorithm in recent multimodal training pipelines, this project explores an alternative: [Reinforce++](https://arxiv.org/pdf/2501.03262v1). The goal is not to replace GRPO, but to investigate how different policy-gradient methods perform on vision-language reasoning tasks.

We evaluate Reinforce++ on two datasets:

- [**THU-KEG/MM_Math**](https://huggingface.co/datasets/THU-KEG/MM_Math): a text-heavy math dataset accompanied by figures.  
- [**hiyouga/geometry3k**](https://huggingface.co/datasets/hiyouga/geometry3k): a geometry-focused benchmark requiring visual understanding.

Each sample follows the format:

```json
{
  "question": "<image>In the figure, $\\overline{A D}$ is perpendicular to $\\overline{B C}$ and $\\overline{A B}$ is perpendicular to $\\overline{A C}$. What is $B C ?$",
  "answer": "20",
  "solution": "\\boxed{20}",
  "id": "geometry3k_2948",
  "image": "geometry3k_images/geometry3k_2948.png",
}
```

📌 Field descriptions:
```
question: mathematical question.

answer: ground-truth numeric answer.

solution: final boxed output with or without step-by-step reasoning.

image: felative path to the image file (relative to the .jsonl location).
```

## 🚀 Quick Start

### 🛠 Installation

Follow the steps below to set up the environment:

```bash
# Conda environment setup
conda create -n maye python=3.11
source activate maye

# PyTorch installation (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# FlashAttention installation (for efficient attention)
pip uninstall -y ninja && pip install ninja
pip install flash-attn --no-build-isolation

# Install other Python dependencies
pip install -r requirements.txt

# Install the project as a package
poetry install

# Set up Weights & Biases for experiment logging
wandb login
```

### 📥 Download Dataset

The paper experiments are conducted on two datasets: **mm_math5k** and **geometry3k**.  
We release them on 🤗 [**ManTle/MAYE**](https://huggingface.co/datasets/ManTle/MAYE).

After downloading, place them under the `datasets` directory with the following structure:

The datasets structure should look like this:
```
.
├── geometry3k
│   ├── geometry3k_images
│   ├── geometry3k_rl_v0.jsonl
│   ├── geometry3k_test_v0.jsonl
│   └── geometry3k_validation_v0.jsonl
└── mm_math
    ├── images
    ├── mathverse_images
    ├── math_verse_test_v0.jsonl
    ├── mm_math_rl_v0.jsonl
    └── mm_math_validation_v0.jsonl

```
🙏 **Acknowledgements**:  
- **mm_math** is based on [THU-KEG/MM_Math](https://huggingface.co/datasets/THU-KEG/MM_Math)
- **geometry3k** is based on [InterGPS](https://github.com/lupantech/InterGPS) and [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)

### 📦 Download Model

The experiments use the instruction-tuned variants of **Qwen2/2.5-VL** (≤7B).  
Please download the following models from Hugging Face and place them in your local directory:

- 🤗[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- 🤗[Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- 🤗[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- 🤗[Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

### 🚀 Launch Training

To launch distributed training, edit the script [`scripts/train_ppo_vllm_distributed.sh`](scripts/train_ppo_vllm_distributed.sh) with the appropriate dataset and model settings. Below is a simplified example:

```bash
# === Basic Configuration ===
DATA=mmmath          # Options: mmmath, geo3k
MODEL=2.5it          # Options: 2.5it (Qwen2.5-VL-Instruct), 2it (Qwen2-VL-Instruct)

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


# === Paths ===
MODEL_PATH=/tmp/ckpts/$MODEL_NAME
PROCESSOR_PATH=/tmp/ckpts/$PROCESSOR_NAME
OUTPUT_DIR=/tmp/ckpts/$MODEL_NAME-$DATADIR_NAME
TRAIN_DATASET_PATH=/tmp/MAYE/datasets/$DATADIR/${TRAIN_DATASET_NAME}.jsonl
VALIDATION_DATASET_PATH=/tmp/MAYE/datasets/$DATADIR/${VALIDATION_DATASET_NAME}.jsonl
TEST_DATASET_PATH=/tmp/MAYE/datasets/$DATADIR/${TEST_DATASET_NAME}.jsonl
```

The script automatically selects the proper dataset and model classes based on your DATA and MODEL choices.
You may then launch training by running:
```bash
bash scripts/train_ppo_vllm_distributed.sh
```
📌 Note: Ensure all paths and checkpoint names are consistent with your local setup.

## 🧷 Citation

If you find our work helpful, please consider citing:

```
@misc{ma2025rethinking,
  title = {Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme},
  author = {Ma, Yan and Chern, Steffi and Shen, Xuyang and Zhong, Yiran and Liu, Pengfei},
  year = {2025},
  howpublished = {\url{https://github.com/GAIR-NLP/MAYE}}
}
```

The arXiv version is currently on hold 😅

