# Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[CONDA_PATH]/lib

# Initialize conda environment
source [CONDA_PATH]/bin/activate
conda init
conda activate [ENV_NAME]

export AGENT_CONFIG='configs/agent_config.yaml' 
python data_generation/dpo_gta_traj/main.py \
        --engine qwen \
        --lora-path '<path to lora weights>' \
        --dpo-agent \
        --sample 3 \
        --verifier 'best_selector' \
        --save-path 'data_generation/dpo_gta_traj/save/preference_traj'   