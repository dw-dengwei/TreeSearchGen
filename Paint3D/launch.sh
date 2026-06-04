#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# for i in {13..13}
# do
#   python pipeline_paint3d_stage1.py \
#    --sd_config controlnet/config/depth_based_inpaint_template.yaml \
#    --render_config paint3d/config/train_config_paint3d.py \
#    --mesh_path demo/objs/Suzanne_monkey/Suzanne_monkey.obj \
#    --prompt " " \
#    --ip_adapter_image_path style_prompt/$i.jpg \
#    --outdir style_output_origin/style_$i/img_stage1
# done


python pipeline_paint3d_stage1.py \
 --sd_config controlnet/config/depth_based_inpaint_template.yaml \
 --render_config paint3d/config/train_config_paint3d.py \
 --mesh_path style_txt/0_2025-04-28_15:42_large_bedroom/decompose/bed.obj \
 --outdir style_txt/0_2025-04-28_15:42_large_bedroom/new_texture/img_stage1 \