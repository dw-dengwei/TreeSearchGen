#!/bin/bash
export VL_MODEL_KEY='empty'
export VL_MODEL_NAME='name'
export VL_MODEL_URL='host'

export LANGUAGE_MODEL_KEY='empty'
export LANGUAGE_MODEL_NAME='name'
export LANGUAGE_MODEL_URL='host'

export HF_ENDPOINT="https://hf-mirror.com"


solver="mcts"
PROJECT_ROOT="output"
BENCHMARK_INSTRUCTIONS="all_instructions.txt"
BLENDER_PATH="CUDA_VISIBLE_DEVICES=0 /home/dw/.local/blender-3.3.21-linux-x64/blender"

mkdir -p "${PROJECT_ROOT}"
set -ex
python generate_layout.py \
    --benchmark_instructions "${BENCHMARK_INSTRUCTIONS}" \
    --output_root "${PROJECT_ROOT}" \
    --furniture_resolution 0.3 --object_resolution 0.1 \
    --tree_search_config "config/config.yaml" \
    --start_step 0 --end_step 17 --use_solver "${solver}" \
    --blender_path "${BLENDER_PATH}" \
    --max_parallel 1 \
    --process_id "0" \
    --prm_threshold 0.3

# step 0-11
# step_0, # Get Room Size
# step_1, # Get Regions
# step_2, # Get Anchor Region
# step_2_3, # Get Region Layout
# step_4, # Get Ground Objects
# step_5, # Get Anchor Objects
# step_6, # Get Group
# step_7, # Get Anchor Placement
# step_8, # Get Other Placement
# step_9, # Get Small Objects Placement
# step_10, # Retrieve Furniture
# step_11, # Get Style

# step 12-13
# step_12, # Furniture Layout
# step_13, # Object Layout

# step 14-15, MAX_PARALLEL=8
# step_14, # Blender Compose
# step_15, # Blender Decompose

# step 16, run_multiple
# step_16, # Re-style objects

# step 17, MAX_PARALLEL=8
# step_17, # Re-Compose Scene
