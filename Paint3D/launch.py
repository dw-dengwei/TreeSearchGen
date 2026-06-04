import os
import argparse
import json
from glob import glob

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def parse():
  argparser = argparse.ArgumentParser(description="Launch a training job.")
  argparser.add_argument(
    "--project_root",
    type=str,
    required=True,
    help="path to project root"
  )
  # argparser.add_argument(
  #   "--scene_root",
  #   type=str,
  #   required=True,
  #   help="Path to the config file.",
  # )
  return argparser.parse_args()

if __name__ == "__main__":
  args = parse()
  path_list = os.listdir(args.project_root)

  path_list.sort(key = lambda x: int(x.split('_')[0]))
  print(path_list)
  for scene_dir in path_list[1:]:
    scene_root = os.path.join(args.project_root, scene_dir)
    if not os.path.exists(os.path.join(scene_root, "14_scene.glb")):
      print(f"Scene {scene_dir} not found, skipping...")
      continue

    objs = glob(os.path.join(scene_root, "decompose", "*.obj"))
    data = json.load(open(os.path.join(scene_root, "14_retrieved_results_with_style.json"), "r"))
    for obj in objs:
      style = ""
      for item in data['objects']:
        # print(list(item.keys())[0], os.path.basename(obj).replace('_', '#').replace('.obj', ''))
        if list(item.keys())[0] == os.path.basename(obj).replace('_', '#').replace('.obj', ''):
          style = item[os.path.basename(obj).replace('_', '#').replace('.obj', '')]["style"]

      if not os.path.exists(f"""{os.path.join(scene_root, "decompose")}/new_texture/{os.path.basename(obj).replace('_', '#').replace('.obj', '')}/txt_stage1/res-0/albedo.png"""):
        prompt_stage1 = f'turn around, {style}, (Sci-Fi digital painting:1.5), colorful, painting, high quality'
        cmd_stage1 = \
      f"""
      python pipeline_paint3d_stage1.py \
      --sd_config controlnet/config/depth_based_inpaint_template.yaml \
      --render_config paint3d/config/train_config_paint3d.py \
      --mesh_path '{obj}' \
      --outdir '{os.path.join(scene_root, "decompose")}/new_texture/{os.path.basename(obj).replace('_', '#').replace('.obj', '')}/txt_stage1' \
      --prompt '{prompt_stage1}'
      """
        print(cmd_stage1)
        os.system(cmd_stage1)

      if not os.path.exists(f"""{os.path.join(scene_root, "decompose")}/new_texture/{os.path.basename(obj).replace('_', '#').replace('.obj', '')}/txt_stage2/tile_res_0/albedo.png"""):
        prompt_stage2 = f"UV map, {style}, Sci-Fi digital painting, high quality"
        cmd_stage2 = \
      f"""
      python pipeline_paint3d_stage2.py \
      --sd_config controlnet/config/UV_based_inpaint_template.yaml \
      --render_config paint3d/config/train_config_paint3d.py \
      --mesh_path '{obj}' \
      --texture_path '{os.path.join(scene_root, "decompose")}/new_texture/{os.path.basename(obj).replace('_', '#').replace('.obj', '')}/txt_stage1/res-0/albedo.png' \
      --outdir '{os.path.join(scene_root, "decompose")}/new_texture/{os.path.basename(obj).replace('_', '#').replace('.obj', '')}/txt_stage2' \
      --prompt '{prompt_stage2}'
      """
        print(cmd_stage2)
        os.system(cmd_stage2)
