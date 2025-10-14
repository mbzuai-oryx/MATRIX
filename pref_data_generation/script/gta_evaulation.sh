 
# Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[CONDA_PATH]/lib

# Initialize conda environment
source [CONDA_PATH]/bin/activate
conda init
conda activate [ENV_NAME]

# Set configuration path
export AGENT_CONFIG='configs/agent_config.yaml'

# Run evaluation script with parameters
python examples/gta/main.py \
    --engine qwen \
    --lora-path [LORA_CHECKPOINT_PATH] \
