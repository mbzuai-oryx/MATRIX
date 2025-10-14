# 🎯 MATRIX – Stage 2: Pref-X Dataset & Preference Optimization (DPO)
🔥 **Preprint 2025 (arXiv: [2510.08567](https://arxiv.org/abs/2510.08567))**

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.08567-b31b1b.svg)](https://arxiv.org/abs/2510.08567)
[![Project Page](https://img.shields.io/badge/🌐-Project_Page-2ea44f.svg)](https://tajamulashraf.com/matrix)
[![GitHub](https://img.shields.io/badge/💻-Code-black.svg)](https://github.com/mbzuai-oryx/MATRIX)
[![Dataset](https://img.shields.io/badge/🤗-MTRACE-blue.svg)](https://huggingface.co/datasets/mbzuai/M-TRACE)

</div>

---

## 🚀 Overview

**Pref-X** refines the **MATRIX** controller beyond imitation.  
It introduces **step-wise preference optimization** through **Direct Preference Optimization (DPO)** to teach agents how to:
- Compare multiple candidate actions,  
- Select the most accurate and consistent, and  
- Learn robust, semantically aligned behavior without explicit rewards.

---

## 🧩 Pref-X Dataset

- **11 K** step-wise preference pairs  
- Generated via exploration + verification pipeline  
- Enables reinforcement-free fine-grained preference tuning  
- Each pair includes one *preferred* and several *dispreferred* actions evaluated by an LLM-based verifier

### 📁 Download Image Captions and Embeddings
Image captions and embeddings can be downloaded here:  
[📦 Google Drive](https://drive.google.com/drive/folders/1Ek6qfmhcaTd7zTEQcBvELh6i7unVhTrk?usp=sharing)

Place the files in: data_generation/sharegpt4v/


Organize as recommended by [ShareGPT4V](https://sharegpt4v.github.io/):
```none
data_generation/sharegpt4v/
├── llava/llava_pretrain/images
├── coco/train2017
├── sam/images
├── web-celebrity/images
├── web-landmark/images
├── wikiart
├── share_textvqa/images
└── chatqa/train/png

```bash
# 1️⃣ Query Generation
# Generate initial task queries.
python data_generation/gta_pipeline/gta0_query_generation.py

# 2️⃣ Image Content Generation
# Convert textual queries to multimodal content descriptions.
python data_generation/gta_pipeline/gta1_query2image_content_parallel.py

# 3️⃣ Image Retrieval
# Fetch images corresponding to generated content.
python data_generation/gta_pipeline/gta2_image_content2image_file.py

# 4️⃣ Quality Filtering
# Verify and retain only high-quality query-image pairs.
python data_generation/gta_pipeline/gta3_q_f_filter_parallel.py


# Collect step-level rollouts for preference comparison.
bash script/trajectory_sampling.sh


# Reformat collected data for DPO training.
python data_generation/dpo_gta_traj/data_reformat/data_formating.py


# Stage 2: Preference Optimization
# Train MATRIX with Direct Preference Optimization using LLaMA-Factory.
bash scripts/train_prefx.sh


# Download the GTA dataset from Hugging Face and place it in:
# data/gta_dataset/

# Run evaluation.
bash script/gta_evaulation.sh

# Results (accuracy, reasoning consistency, and tool-use success)
# are stored in the .cache/ directory.
