<div align="center">

<h2>Global-Local Tree Search in VLMs for 3D Indoor Scene Generation (CVPR 2025)</h2>
<div>
    <a href='https://dw-dengwei.cn/' target='_blank'>Wei Deng</a>&emsp;
    <a href='https://jueduilingdu.github.io/' target='_blank'>Mengshi Qi</a><sup>*</sup>&emsp;
    Huadong Ma&emsp;
</div>
<p>
    <a href='https://cvpr.thecvf.com/'><img src='https://img.shields.io/badge/CVPR-2025-blue.svg' alt='CVPR 2025'></a>
    <a href='https://arxiv.org/abs/2503.16113'><img src='https://img.shields.io/badge/arXiv-2503.16113-b31b1b.svg' alt='arXiv 2025'></a>
    <a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache%202.0-green.svg' alt='Apache 2.0'></a>
</p>

> This paper considers 3D indoor scene generation as a planning problem subject to spatial and layout common sense constraints. To solve the problem with a VLM, we propose a new global-local tree search algorithm that decomposes the generation process into hierarchical planning stages, leveraging tree search to explore and optimize the layout at both global room-level and local object-level granularity.


<h2>Global-Local Monte Carlo Tree Search in Vision-Language Models for Text-to-3D Indoor Scene Generation (arXiv 2026)</h2>
<div>
    <a href='https://jueduilingdu.github.io/' target='_blank'>Mengshi Qi</a><sup>*</sup>&emsp;
    <a href='https://dw-dengwei.cn/' target='_blank'>Wei Deng</a>&emsp;
    Xianlin Zhang&emsp;
    Huadong Ma&emsp;
</div>
<p>
    <a href='https://arxiv.org/abs/2606.06002'><img src='https://img.shields.io/badge/arXiv-2606.06002-b31b1b.svg' alt='arXiv 2026'></a>
    <a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache%202.0-green.svg' alt='Apache 2.0'></a>
</p>
</div>

> This paper is an extension of our CVPR 2025 paper. We integrate PRM-guided MCTS for tree search and a new re-texture pipeline, and propose a pipeline, 3DTindo-Bench for text-to-3D scene generation.

## 🛠️ Setup

### Requirements

- Python >= 3.12
- Blender 3.3+
- CUDA 11.8
- [uv](https://docs.astral.sh/uv/)

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Clone the repository
git clone https://github.com/dw-dengwei/TreeSearchGen.git
cd TreeSearchGen

# Create a virtual environment and install dependencies
uv sync
```

### Database

Please follow [holodeck](https://github.com/allenai/Holodeck) to download 3D assets.

### Environment Variables

Set these before running:

```bash
# Vision-Language Model API
export VL_MODEL_KEY='your_api_key'
export VL_MODEL_NAME='model_name'
export VL_MODEL_URL='api_endpoint'

# Language Model API
export LANGUAGE_MODEL_KEY='your_api_key'
export LANGUAGE_MODEL_NAME='model_name'
export LANGUAGE_MODEL_URL='api_endpoint'

# Optional: HuggingFace mirror (for users in China)
export HF_ENDPOINT="https://hf-mirror.com"


BLENDER_PATH="CUDA_VISIBLE_DEVICES=0 /path/to/blender"
```

## 🚀 Usage

### Quick Start

```bash
bash run.sh
```

The script in `run.sh` runs the full pipeline:
- Solver: `mcts` (tree search)
- Instructions loaded from `all_instructions.txt`
- Output directory: `./output/`
- Parallel execution configurable via `--max_parallel`

### Advanced Usage

```bash
python generate_layout.py \
    --benchmark_instructions "all_instructions.txt" \
    --output_root "output" \
    --furniture_resolution 0.3 \
    --object_resolution 0.1 \
    --tree_search_config "config/config.yaml" \
    --start_step 0 \
    --end_step 17 \
    --use_solver "mcts" \
    --blender_path "blender" \
    --max_parallel 4 \
    --process_id "0" \
    --prm_threshold 0.3
```

| Argument | Description |
|----------|-------------|
| `--start_step` / `--end_step` | Run a subset of the pipeline (e.g., `--start_step 12 --end_step 13` for layout only) |
| `--use_solver` | Search algorithm: `mcts` (default) or `dfs` |
| `--max_parallel` | Maximum parallel instructions |
| `--process_id` | Process range (e.g., `"0-5"`, `"0,2,4"`) for distributed execution |
| `--furniture_resolution` | Grid resolution for furniture layout (meters) |
| `--object_resolution` | Grid resolution for small object layout (meters) |
| `--prm_threshold` | Threshold for the Process Reward Model |

### Re-texturing with Paint3D

We apply texture refinement using [Paint3D](https://github.com/OpenTexture/Paint3D).
Please download the checkpoints.

## 📊 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{deng2025global,
  title={Global-Local Tree Search in VLMs for 3D Indoor Scene Generation},
  author={Deng, Wei and Qi, Mengshi and Ma, Huadong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@article{qi2026global,
  title={Global-Local Monte Carlo Tree Search in Vision-Language Models for Text-to-3D Indoor Scene Generation},
  author={Qi, Mengshi and Deng, Wei and Zhang, Xianlin and Ma, Huadong},
  journal={arXiv preprint arXiv:2606.06002},
  year={2026}
}
```

## 🙏 Acknowledgement

We thank the authors of [Objaverse](https://objaverse.allenai.org/) for providing the 3D model database, [Paint3D](https://github.com/OpenTexture/Paint3D) for the texture refinement pipeline, and [Blender](https://www.blender.org/) for the rendering engine.
