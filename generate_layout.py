import os
import argparse
import json
import matplotlib.pyplot as plt
import traceback
import time
import yaml
import asyncio
from utils.logger import logger
from glob import glob

from colorama import Fore
from tqdm.auto import tqdm
from utils.post_process import process
from utils.consistent import get_consistent_response
from utils.get_qwen import get_response_fn, get_api
from easydict import EasyDict as edict

from agent.small_object_agent_new import SmallObjAgent
from agent.room_agent_new import RoomAgent
from agent.generate_supported import get_supported
from agent.region import get_regions
from agent.anchor_select import get_anchor
from agent.anchor_region_select import get_anchor_region
from agent.room_size import get_room_size
from agent.gen_region_layout import get_region_layout
from agent.anchor_object_layout import get_anchor_placement_attr
from agent.other_object_layout import get_other_placement_attr
from agent.small_objects import get_small_objects
from agent.object_appearance import get_appearance_desc

  
vl_model_name = os.environ['VL_MODEL_NAME']
vl_model_key = os.environ['VL_MODEL_KEY']
vl_model_url = os.environ['VL_MODEL_URL']

language_model_name = os.environ['LANGUAGE_MODEL_NAME']
language_model_key = os.environ['LANGUAGE_MODEL_KEY']
language_model_url = os.environ['LANGUAGE_MODEL_URL']


language_model_api = get_api(api_key=language_model_key, api_url=language_model_url)
vl_model_api = get_api(api_key=vl_model_key, api_url=vl_model_url)


class TreeSearchGen:
  def __init__(self, output_root, tree_search_config, furniture_resolution, object_resolution, use_solver, visualize_mcts, benchmark_instructions, start_step, end_step, blender_path, prm_threshold, max_parallel=4, use_image=True):
    self.output_root = output_root
    self.tree_search_config = tree_search_config
    self.furniture_resolution = furniture_resolution
    self.object_resolution = object_resolution
    self.use_solver = use_solver
    self.visualize_mcts = visualize_mcts
    self.benchmark_instructions = benchmark_instructions
    self.start_step = start_step
    self.end_step = end_step
    self.blender_path = blender_path
    self.prm_threshold = prm_threshold
    self.failed_instructions = []  # 记录失败的指令
    self.max_retries = 1  # 最大重试次数
    self.max_parallel = max_parallel  # 最大并行数量
    self.use_image = use_image
    
    # 如果step 16在执行范围内，才导入Paint3D模块
    if start_step <= 16 <= end_step:
      os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
      from Paint3D.pipeline_paint3d_stage1 import retexture_stage1
      from Paint3D.pipeline_paint3d_stage2 import retexture_stage2
      self.retexture_stage1 = retexture_stage1
      self.retexture_stage2 = retexture_stage2
    
    if start_step <= 10 <= end_step:
      from retrieve import retrieve_furniture
      self.retrieve_furniture = retrieve_furniture
    
    self.failure_log = os.path.join(self.output_root, 'failed_instructions.json')
    if not start_step >= 14:
      # 检查失败日志文件是否存在
      if os.path.exists(self.failure_log):
        past_error = json.load(open(self.failure_log))
        if len(past_error) > 0:
          pre_failed_instructions = json.dumps(past_error)
          logger.error(f"Failure log file already exists: {pre_failed_instructions}")
          raise FileExistsError(f"Failure log file already exists: {self.failure_log}")
    
    # 创建空的失败日志文件
    json.dump([], open(self.failure_log, 'w'), indent=2)

    # 初始化语言模型和视觉语言模型的基本配置
    self.language_model_base = {
      'inference_fn': language_model_api,
      'is_vl_model': False,
      'model': language_model_name,
      'max_retry': 5,
      'system_prompt': 'You are a helpful AI assistant.'
    }

    self.vl_model_base = {
      'inference_fn': vl_model_api,
      'is_vl_model': True,
      'model': vl_model_name,
      'max_retry': 2,
      'system_prompt': 'You are a helpful AI assistant.'
    }

  def get_instance_specific_models(self, log_name):
    """为每个实例创建独立的模型实例"""
    language_model = get_response_fn(
      **self.language_model_base,
      verbose=log_name
    )

    vl_model = get_response_fn(
      **self.vl_model_base,
      verbose=log_name
    )
    
    return language_model, vl_model

  def execute_with_retry(self, step_func, step_name):
    """执行步骤函数，支持重试机制
    以下错误类型不进行重试，直接返回失败：
    - KeyError: 字典键不存在或JSON解析错误
    - FileNotFoundError: 文件不存在
    - UnboundLocalError: 局部变量未定义
    其他错误类型会进行重试
    """
    retries = 0
    last_error = None
    while retries < self.max_retries:
      try:
        step_func()
        return True, None
      except (KeyError, FileNotFoundError, UnboundLocalError) as e:
        # 这些错误直接返回失败，不进行重试
        last_error = traceback.format_exc()
        error_type = type(e).__name__
        logger.error(f"Step {step_name} failed with {error_type}: {last_error}")
        return False, last_error
      except Exception as e:
        retries += 1
        last_error = traceback.format_exc()
        logger.error(f"Step {step_name} failed (attempt {retries}/{self.max_retries}): {last_error}")
        if retries == self.max_retries:
          logger.error(f"Step {step_name} failed after {self.max_retries} attempts")
          return False, last_error
        time.sleep(1)  # 重试前等待1秒
    return False, last_error

  # def set_run(self, index, instruction):
  #   self.description = instruction
  #   self.output_dir = os.path.join(self.output_root, f'{index}')
  #   os.makedirs(self.output_dir, exist_ok=True)
  #   self.log_name = os.path.join(self.output_dir, 'log.ansi')
  #   self.language_model = get_response_fn(
  #     language_model_api,
  #     is_vl_model=False,
  #     model=language_model_name,
  #     max_retry=5,
  #     verbose=self.log_name,
  #     system_prompt='You are a helpful AI assistant.'
  #   )

  #   self.vl_model = get_response_fn(
  #     vl_model_api,
  #     is_vl_model=True,
  #     model=vl_model_name,
  #     max_retry=2,
  #     verbose=self.log_name,
  #     system_prompt='You are a helpful AI assistant.'
  #   )

  @staticmethod
  def step_0(cls, context):
    room_topo = {}
    room_dimension_output = get_room_size(context['description'], context['language_model']('creative'))['output']
    room_topo['room_dimension'] = room_dimension_output['size']
    room_topo['room_desc'] = context['description']
    room_topo['functional_area'] = []

    json.dump(room_topo, open(os.path.join(context['output_dir'], '1_room_size.json'), 'w'), indent=2)


  @staticmethod
  def step_1(cls, context):
    room_topo = json.load(open(os.path.join(context['output_dir'], '1_room_size.json')))
    region_output = get_regions(context['description'], context['language_model']('creative'))['output']
    for r in region_output['functional_zones']:
      if process(r['name']) not in [process(_['name']) for _ in room_topo['functional_area']]:
        room_topo['functional_area'].append({
          "name": process(r['name']),
          "desc": r['description']
        })

    json.dump(room_topo, open(os.path.join(context['output_dir'], '2_regions.json'), 'w'), indent=2)

  @staticmethod
  def step_2(cls, context):
    pass


  @staticmethod
  def step_2_3(cls, context):
    room_topo = json.load(open(os.path.join(context['output_dir'], '2_regions.json')))
    areas = []
    for a in room_topo['functional_area']:
      areas.append(a)
    anchor_region_output, _ = get_consistent_response(
      fn=get_anchor_region,
      args=(context['description'], areas, context['language_model']('precise')),
      keys='anchor_region_name',
      times=1,
    )

    json.dump(room_topo, open(os.path.join(context['output_dir'], '3_anchor_region.json'), 'w'), indent=2)
    
    for idx, a in enumerate(room_topo['functional_area']):
      room_topo['functional_area'][idx]['related_anchor_region'] = process(anchor_region_output)

    region_layout_output = get_region_layout(
      ("\n".join([f"  - {area['name']}: {area['desc']}" for area in room_topo['functional_area']]), [area['name'] for area in room_topo['functional_area']], len(room_topo['functional_area'])), room_topo['functional_area'][0]['related_anchor_region'],
      context['description'], room_topo['room_dimension'], context['language_model']('creative')
    )['output']
    rank_to_name = [
      None,
      {1: 'center'},
      {1: 'left side', 2: 'right side'},
      {1: 'left side', 2: 'center', 3: 'right side'},
      {1: 'left side', 2: 'left center', 3: 'right center', 4: 'right side'},
      {1: 'left side', 2: 'left center', 3: 'center', 4: 'right center', 5: 'right side'}
    ]
    for idx, area in enumerate(room_topo['functional_area']):
      changed = False
      for output in region_layout_output:
        N_regions = len(room_topo['functional_area'])
        if output['region_name'].lower() == area['name'].lower():
          room_topo['functional_area'][idx]['region_room_relation'] = rank_to_name[N_regions][int(output['region_room_relation'])]
          room_topo['functional_area'][idx]['region_dimension'] = output['region_dimension']
          changed = True
          break
      assert changed
    room_topo['room_dimension'][0] = sum([a['region_dimension'][0] for a in room_topo['functional_area']])

    json.dump(room_topo, open(os.path.join(context['output_dir'], '4_region_layout.json'), 'w'), indent=2)


  @staticmethod
  def step_4(cls, context):
    room_topo = json.load(open(os.path.join(context['output_dir'], '4_region_layout.json')))
    supporting = 'floor'
    fur_list = []
    for idx, r in enumerate(tqdm(room_topo['functional_area'], desc='Area', leave=False)):
      area_desc = r['desc']
      area_dimension = r['region_dimension']
      furniture_output: dict = get_supported(r, area_dimension, context['language_model']('precise'), None)['output']
      furniture_output.pop('input')
      furniture_output.pop('raw')
      for fur_idx, fur in enumerate(furniture_output['furnitures']):
        furniture_output['furnitures'][fur_idx]['name'] = process(fur['name'])
        fur_list.append(process(fur['name']))
      room_topo['functional_area'][idx].update(furniture_output)

    fur_count = {}
    for idx, r in enumerate(room_topo['functional_area']):
      for fur_idx, fur in enumerate(r['furnitures']):
        name = room_topo['functional_area'][idx]['furnitures'][fur_idx]['name']
        if name not in fur_count.keys():
          fur_count[name] = 1
        else:
          fur_count[name] += 1
        if fur_list.count(name) >= 2:
          name = f"{name}#{fur_count[name]}"
        room_topo['functional_area'][idx]['furnitures'][fur_idx]['name'] = name

    json.dump(room_topo, open(os.path.join(context['output_dir'], '5_ground_objects.json'), 'w'), indent=2)


  @staticmethod
  def step_5(cls, context):
    room_topo = json.load(open(os.path.join(context['output_dir'], '5_ground_objects.json')))
    for idx, r in enumerate(room_topo['functional_area']):
      furnitures_str = []
      furnitures = []
      for item_idx, item in enumerate(r['furnitures']):
        furnitures.append(item['name'])
        furnitures_str.append(
          f"  {item_idx + 1}. {item['name']}: {item['description']}"
        )
      furnitures_str = ('\n'.join(furnitures_str), len(furnitures_str), furnitures)
      area = r['area']
      anchor_output, _ = get_consistent_response(
        fn=get_anchor,
        args=(context['description'], area, furnitures_str, context['language_model']('precise')),
        keys='anchor_name',
        times=1,
      )
      for f_idx, item in enumerate(r['furnitures']):
        name = item['name']
        if name == anchor_output:
          room_topo['functional_area'][idx]['furnitures'][f_idx]['type'] = 'anchor'
        else:
          room_topo['functional_area'][idx]['furnitures'][f_idx]['type'] = 'other'

    json.dump(room_topo, open(os.path.join(context['output_dir'], '6_anchor_object.json'), 'w'), indent=2)


  @staticmethod
  def step_6(cls, context):
    room_topo = json.load(open(os.path.join(context['output_dir'], '6_anchor_object.json')))
    for idx, r in enumerate(room_topo['functional_area']):
      anchor_furniture_str = []
      other_furnitures_str = []
      anchor_furniture_list = []
      other_furniture_list = []
      area = r['area']
      for item in r['furnitures']:
        if item['type'].lower() == 'anchor':
          anchor_furniture_str.append(
            f"- {item['name']}\n  - \"description\": {item['description']}"
          )
          anchor_furniture_list.append(item['name'])
        elif item['type'].lower() == 'other':
          other_furnitures_str.append(
            f"- {item['name']}\n  - \"description\": {item['description']}"
          )
          other_furniture_list.append(item['name'])
      anchor_furniture_str = ('\n'.join(anchor_furniture_str), len(anchor_furniture_str), anchor_furniture_list)
      other_furnitures_str = ('\n'.join(other_furnitures_str), len(other_furnitures_str), other_furniture_list)

      group_output = []
      for item in r['furnitures']:
        group_output.append({
          "name": item['name'],
          "description": item['description'],
          "related_anchor_object": anchor_furniture_list[0]
        })

      for f_idx, item in enumerate(r['furnitures']):
        type = item['type']
        name = item['name']
        if type.lower() == 'anchor':
          room_topo['functional_area'][idx]['furnitures'][f_idx]['related_anchor_object'] = name
        elif type.lower() == 'other':
          for out_item in group_output:
            if out_item['name'] == name:
              room_topo['functional_area'][idx]['furnitures'][f_idx]['related_anchor_object'] = out_item['related_anchor_object']
              break

    json.dump(room_topo, open(os.path.join(context['output_dir'], '7_group.json'), 'w'), indent=2)


  @staticmethod
  def step_7(cls, context):
    room_topo = json.load(open(os.path.join(context['output_dir'], '7_group.json')))
    for idx, area in enumerate(room_topo['functional_area']):
      anchor_idx = 9999
      for t, obj in enumerate(area['furnitures']):
        if obj['type'].lower() == 'anchor':
          anchor_name = obj['name']
          anchor_desc = obj['description']
          anchor_idx = t
          break

      anchor_placement_output = get_anchor_placement_attr(
        area['name'], area['desc'], area['region_dimension'],
        anchor_name, anchor_desc, context['language_model']('precise')
      )['output']
      room_topo['functional_area'][idx]['furnitures'][anchor_idx]['placement_rule'] = anchor_placement_output['placement']

    json.dump(room_topo, open(os.path.join(context['output_dir'], '8_anchor_placement.json'), 'w'), indent=2)


  @staticmethod
  def step_8(cls, context):
    room_topo = json.load(open(os.path.join(context['output_dir'], '8_anchor_placement.json')))
    for area_idx, area in enumerate(room_topo['functional_area']):
      for fur_idx, obj in enumerate(area['furnitures']):
        if obj['type'].lower() == 'anchor':
          anchor_name = obj['name']
          anchor_desc = obj['description']

      for fur_idx, obj in enumerate(area['furnitures']):
        if obj['type'].lower() != 'anchor' and 'placement_rule' not in obj.keys():
          other_placement_output, _ = get_consistent_response(
            fn=get_other_placement_attr,
            args=(area['name'], area['desc'], anchor_name, anchor_desc, obj['name'], obj['description'], context['description'], context['language_model']('precise')),
            keys=['distance', 'placement_rule', 'functionally_grouped', 'close_to_the_wall', 'alignment'],
            times=1
          )
          room_topo['functional_area'][area_idx]['furnitures'][fur_idx]['distance'] = other_placement_output['distance']
          room_topo['functional_area'][area_idx]['furnitures'][fur_idx]['placement_rule'] = other_placement_output['placement_rule']
          room_topo['functional_area'][area_idx]['furnitures'][fur_idx]['functionally_grouped'] = other_placement_output['functionally_grouped']
          room_topo['functional_area'][area_idx]['furnitures'][fur_idx]['close_to_the_wall'] = other_placement_output['close_to_the_wall']
          room_topo['functional_area'][area_idx]['furnitures'][fur_idx]['alignment'] = other_placement_output['alignment']
          
      json.dump(room_topo, open(os.path.join(context['output_dir'], '9_other_placement.json'), 'w'), indent=2)
    

  @staticmethod
  def step_9(cls, context):
    room_topo = json.load(open(os.path.join(context['output_dir'], '9_other_placement.json')))
    furniture_list = []
    for area_idx, area in enumerate(room_topo['functional_area']):
      for fur_idx, fur in enumerate(area['furnitures']):
        name = fur['name']
        desc = fur['description']
        size = fur['size']
        furniture_list.append({
          'name': name,
          'description': desc,
          'size': size,
        })
    furniture_input = json.dumps(furniture_list)
    output = get_small_objects(furniture_input, context['language_model']('creative'))['output']
    room_topo['small_objects'] = output

    json.dump(room_topo, open(os.path.join(context['output_dir'], '10_full_scene_graph.json'), 'w'), indent=2)

  @staticmethod
  def step_10(cls, context):
    cls.retrieve_furniture(context['output_dir'], 'top', 0.7)

  @staticmethod
  def step_11(cls, context):
    objects = json.load(open(os.path.join(context['output_dir'], '11_retrieved_results.json')))
    floor_objects = []
    for fur_idx, obj in enumerate(objects['objects']):
      floor_objects.append(list(obj.keys())[0])
    output = get_appearance_desc(floor_objects, context['description'], context['language_model']('creative'))['output']
    objects['style_reasoning'] = output['reasoning']
    objects['floor_texture'] = output['floor_texture']

    for fur_idx, obj in enumerate(objects['objects']):
      name = list(obj.keys())[0]
      for item in output['furniture_list']:
        if item['name'] == name:
          obj[name]['style'] = item['appearance']
          break
    json.dump(objects, open(os.path.join(context['output_dir'], '12_retrieved_results_with_style.json'), 'w'), indent=2)

  @staticmethod
  def step_12(cls, context):
    room_agent = RoomAgent(
      os.path.join(context['output_dir'], '10_full_scene_graph.json'),
      tree_search_config=context['tree_search_config'],
      vl_model=context['vl_model']('precise'),
      language_model=context['language_model']('precise'),
      verbose=False,
      resolution=context['furniture_resolution'],
      use_solver=context['use_solver'],
      visualize_mcts=context['visualize_mcts'],
      output_dir=context['output_dir'],
      instruction=context['description'],
      prm_threshold=context['prm_threshold'],
      use_image=context['use_image']
    )
    results = room_agent.forward()
    img = RoomAgent.visualize_result(results, room_agent.resolution)
    json.dump(results, open(os.path.join(context['output_dir'], '13_furniture_layout.json'), 'w'), indent=2)
    plt.imsave(os.path.join(context['output_dir'], '13_furniture_layout.png'), img)

  @staticmethod
  def step_13(cls, context):
    tmp = json.load(open(os.path.join(context['output_dir'], '13_furniture_layout.json')))
    room_dimension = [0, tmp['room_dimension'][0], 0, tmp['room_dimension'][1]]
    agent = SmallObjAgent(
      context['output_dir'],
      os.path.join(context['output_dir'], '13_furniture_layout.json'),
      os.path.join(context['output_dir'], '10_full_scene_graph.json'),
      tree_search_config=context['tree_search_config'],
      vl_model=context['vl_model']('precise'),
      language_model=context['language_model']('precise'),
      visualize_intermediate=False,
      resolution=context['object_resolution'],
      vis_size=1.2,
      render_size=0.5,
      use_solver=context['use_solver'],
      visualize_mcts=context['visualize_mcts'],
      instruction=context['description'],
      prm_threshold=context['prm_threshold'],
      use_image=context['use_image']
    )
    results = agent.forward()
    img = SmallObjAgent.visualize_result(results, 0.3, room_dimension)
    json.dump(results, open(os.path.join(context['output_dir'], '14_small_object_layout.json'), 'w'), indent=2)
    plt.imsave(os.path.join(context['output_dir'], '14_small_object_layout.png'), img)

  @staticmethod
  def step_14(cls, context):
    # os.system(f"{context['blender_path']} --background --python blender_placement.py -- --write_glb  --project_dir {context['output_dir']} 2>&1")
    os.system(f"{context['blender_path']} --background --python blender_placement.py -- --write_glb  --project_dir {context['output_dir']} > {os.path.join(context['output_dir'], 'blender_placement.log')}")

  @staticmethod
  def step_15(cls, context):
    # os.system(f"{context['blender_path']} --background --python decompose_scene.py -- {context['output_dir']}  2>&1")
    os.system(f"{context['blender_path']} --background --python decompose_scene.py -- {context['output_dir']} > {os.path.join(context['output_dir'], 'blender_decompose.log')}")

  @staticmethod
  def step_16(cls, context):
    scene_root = context['output_dir']
    objs = glob(os.path.join(scene_root, "decompose", "*.obj"))
    data = json.load(open(os.path.join(scene_root, "12_retrieved_results_with_style.json"), "r"))
    for obj in tqdm(objs, desc='Re-style objects', leave=False):
      style = ""
      if os.path.basename(obj).replace('_', ' ').replace('.obj', '') == 'Floor':
        style = data['floor_texture']
      else:
        for item in data['objects']:
          if list(item.keys())[0].replace('_', ' ').replace('#', ' ') == os.path.basename(obj).replace('_', ' ').replace('#', ' ').replace('.obj', ''):
            style = item[list(item.keys())[0]]['name'] + ', ' + item[list(item.keys())[0]].get("style", "")
      
      # if not os.path.exsts(f"""{os.path.join(scene_root, "decompose")}/new_texture/{os.path.basename(obj).replace('_', '#').replace('.obj', '')}/txt_stage1/res-0/albedo.png"""):
      if True:
        prompt_stage1 = f'turn around, {style}, high quality'
        print(prompt_stage1)
        cls.retexture_stage1(
          output_dir=os.path.join(scene_root, "decompose", "new_texture", os.path.basename(obj).replace('_', '#').replace('.obj', ''), 'txt_stage1'),
          prompt=prompt_stage1,
          obj_path=obj
        )
      
      # if not os.path.exists(f"""{os.path.join(scene_root, "decompose")}/new_texture/{os.path.basename(obj).replace('_', '#').replace('.obj', '')}/txt_stage2/res-0/albedo.png"""):
      if True:
        prompt_stage2 = f"UV map, {style}, high quality"
        print(prompt_stage2)
        cls.retexture_stage2(
          output_dir=os.path.join(scene_root, "decompose", "new_texture", os.path.basename(obj).replace('_', '#').replace('.obj', ''), 'txt_stage2'),
          prompt=prompt_stage2,
          obj_path=obj,
          texture_path=os.path.join(scene_root, "decompose", "new_texture", os.path.basename(obj).replace('_', '#').replace('.obj', ''), 'txt_stage1', 'res-0', 'albedo.png'),
        )

  @staticmethod
  def step_17(cls, context):
    os.system(f"{context['blender_path']} --background --python compose_objects.py -- {context['output_dir']} > {os.path.join(context['output_dir'], 'blender_compose_objects.log')} 2>&1")


  async def process_single_instruction(self, index, instruction, opts, step_names):
    """异步处理单个指令"""
    # 创建实例特定的变量
    output_dir = os.path.join(self.output_root, f'{index}')
    os.makedirs(output_dir, exist_ok=True)
    log_name = os.path.join(output_dir, 'log.ansi')
    
    # 为这个实例创建独立的模型
    language_model, vl_model = self.get_instance_specific_models(log_name)
    
    # 创建一个上下文对象来存储实例特定的状态
    context = {
      'description': instruction,
      'output_dir': output_dir,
      'log_name': log_name,
      'language_model': language_model,
      'vl_model': vl_model,
      'tree_search_config': self.tree_search_config,
      'furniture_resolution': self.furniture_resolution,
      'object_resolution': self.object_resolution,
      'use_solver': self.use_solver,
      'visualize_mcts': self.visualize_mcts,
      'blender_path': self.blender_path,
      'prm_threshold': self.prm_threshold,
      'use_image': self.use_image
    }
    
    time_start = time.time()
    failed_steps = []
    
    # 创建进度条
    total_steps = self.end_step - self.start_step + 1
    with tqdm(total=total_steps, desc=f"Instruction {index}", leave=False) as pbar:
      for i in range(self.start_step, self.end_step + 1):
        # 更新进度条描述以显示当前步骤
        pbar.set_description(f"Instruction {index} - Step {i}: {step_names[i]}")
        
        # 创建一个包装函数来处理实例特定的状态
        async def execute_step():
          step_func = opts[i]
          if hasattr(step_func, '__get__'):
            # 如果是绑定方法，创建一个新的绑定到当前实例的方法
            bound_func = step_func.__get__(self, self.__class__)
            return await asyncio.get_event_loop().run_in_executor(
              None,
              lambda: self.execute_with_retry_context(bound_func, context, step_names[i])
            )
          else:
            # 如果是普通函数，直接执行
            return await asyncio.get_event_loop().run_in_executor(
              None,
              lambda: self.execute_with_retry_context(step_func, context, step_names[i])
            )

        success, error = await execute_step()
        
        if not success:
          failed_steps.append({
            'step_index': i,
            'step_name': step_names[i],
            'error': error
          })
          # 立即保存失败信息
          failed_info = {
            'index': index,
            'instruction': instruction,
            'failed_steps': failed_steps
          }
          self.failed_instructions.append(failed_info)
          # 使用异步锁保护文件写入
          async with self._file_lock:
            json.dump(self.failed_instructions, open(self.failure_log, 'w'), indent=2)
          return
        
        # 更新进度条
        pbar.update(1)
    
    time_end = time.time()
    print('Time cost:', time_end - time_start, file=open(log_name, 'a'))

  def execute_with_retry_context(self, func, context, step_name):
    """使用上下文执行步骤函数，支持重试机制"""
    retries = 0
    last_error = None
    while retries < self.max_retries:
      try:
        func(context)
        
        return True, None
      except (KeyError, FileNotFoundError, UnboundLocalError) as e:
        last_error = traceback.format_exc()
        error_type = type(e).__name__
        logger.error(f"Step {step_name} failed with {error_type}: {last_error}")
        return False, last_error
      except Exception as e:
        retries += 1
        last_error = traceback.format_exc()
        logger.error(f"Step {step_name} failed (attempt {retries}/{self.max_retries}): {last_error}")
        if retries == self.max_retries:
          logger.error(f"Step {step_name} failed after {self.max_retries} attempts")
          return False, last_error
    return False, last_error

  def run(self, process_id=None):
    start_time = time.time()  # Start timing
    opts = [
      TreeSearchGen.step_0, # Get Room Size
      TreeSearchGen.step_1, # Get Regions
      TreeSearchGen.step_2, # Get Anchor Region
      TreeSearchGen.step_2_3, # Get Region Layout
      TreeSearchGen.step_4, # Get Ground Objects
      TreeSearchGen.step_5, # Get Anchor Objects
      TreeSearchGen.step_6, # Get Group
      TreeSearchGen.step_7, # Get Anchor Placement
      TreeSearchGen.step_8, # Get Other Placement
      TreeSearchGen.step_9, # Get Small Objects Placement
      TreeSearchGen.step_10, # Retrieve Furniture
      TreeSearchGen.step_11, # Get Style

      TreeSearchGen.step_12, # Furniture Layout
      TreeSearchGen.step_13, # Object Layout

      TreeSearchGen.step_14, # Blender Compose
      TreeSearchGen.step_15, # Blender Decompose
      TreeSearchGen.step_16, # Re-style objects
      TreeSearchGen.step_17, # Re-Compose Scene
    ]
    step_names = [
      "Get Room Size",
      "Get Regions",
      "Get Anchor Region",
      "Get Region Layout",
      "Get Ground Objects",
      "Get Anchor Objects",
      "Get Group",
      "Get Anchor Placement",
      "Get Other Placement",
      "Get Small Objects Placement",
      "Retrieve Furniture",
      "Get Style",
      "Furniture Layout",
      "Object Layout",
      "Blender Compose",
      "Blender Decompose",
      "Re-style objects",
      "Re-Compose Scene",
    ]
    
    # 解析process_id参数
    if process_id is None:
      use_range = True
      start_index = 0
      end_index = None
    else:
      if '-' in process_id:
        start_str, end_str = process_id.split('-')
        start_index = int(start_str)
        end_index = int(end_str) if end_str else None
        use_range = True
      else:
        instance_indices = [int(x.strip()) for x in process_id.split(',')]
        use_range = False

    # 创建要处理的指令列表
    instructions_to_process = []
    for index, instruction in enumerate(self.benchmark_instructions):
      if use_range:
        if index < start_index or (end_index is not None and index >= end_index):
          continue
      else:
        if index not in instance_indices:
          continue
      instructions_to_process.append((index, instruction))

    # 创建事件循环和文件锁
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    self._file_lock = asyncio.Lock()
    
    # 创建信号量来控制并发数量
    semaphore = asyncio.Semaphore(self.max_parallel)

    async def process_with_semaphore(index, instruction):
      async with semaphore:  # 使用信号量控制并发
        return await self.process_single_instruction(index, instruction, opts, step_names)

    # 创建任务列表
    tasks = [
      process_with_semaphore(index, instruction)
      for index, instruction in instructions_to_process
    ]

    # 运行所有任务
    try:
      loop.run_until_complete(asyncio.gather(*tasks))
    finally:
      loop.close()
    
    # 输出失败结果总结
    if self.failed_instructions:
      print(f"\n{Fore.RED}Failed Instructions Summary:{Fore.RESET}")
      for failed in self.failed_instructions:
        print(f"\nInstruction {failed['index']}:")
        print(f"Text: {failed['instruction']}")
        for step in failed['failed_steps']:
          print(f"Failed at step: {step['step_name']}")
          print(f"Error: {step['error']}")
      
      print(f"\nDetailed failure log saved to: {self.failure_log}")
    
    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes). Total instructions: {len(instructions_to_process)}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_root')
  parser.add_argument('--tree_search_config')
  parser.add_argument('--start_step', type=int, default=0)
  parser.add_argument('--end_step', type=int, default=-1)
  parser.add_argument('--process_id', type=str, default='0', help='Process ID specification. For range mode: "1-2" means [1,2), "1-" means [1,infinity); For list mode: "1,2" means [1,2], "1" means [1]')
  parser.add_argument('--furniture_resolution', type=float)
  parser.add_argument('--benchmark_instructions', type=str)
  parser.add_argument('--object_resolution', type=float)
  parser.add_argument('--use_solver', type=str, default='mcts')
  parser.add_argument('--visualize_mcts', action='store_true')
  parser.add_argument('--blender_path', type=str, default='blender')
  parser.add_argument('--max_parallel', type=int, default=4, help='Maximum number of parallel tasks')
  parser.add_argument('--prm_threshold', type=float, default=0.3, help='PRM threshold')
  parser.add_argument('--ablate_image', action='store_true')
  args = parser.parse_args()

  with open(args.benchmark_instructions, 'r') as f:
    benchmark_instructions = f.read().splitlines()

  tree_search_config = edict(yaml.safe_load(open(args.tree_search_config, 'r')))
  ts = TreeSearchGen(
    output_root=args.output_root,
    tree_search_config=tree_search_config,
    furniture_resolution=args.furniture_resolution,
    object_resolution=args.object_resolution,
    use_solver=args.use_solver,
    visualize_mcts=args.visualize_mcts,
    benchmark_instructions=benchmark_instructions,
    start_step=args.start_step,
    end_step=args.end_step,
    blender_path=args.blender_path,
    max_parallel=args.max_parallel,
    prm_threshold=args.prm_threshold,
    use_image=not args.ablate_image
  )

  ts.run(process_id=args.process_id)