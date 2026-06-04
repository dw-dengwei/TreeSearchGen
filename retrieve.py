import json
from tqdm.auto import tqdm
import argparse
from utils.objaverse import ObjathorRetriever, get_bbox_dims
import open_clip
from sentence_transformers import SentenceTransformer
from utils.logger import logger
import os
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k", device=device)
clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
sbert_model = SentenceTransformer("all-mpnet-base-v2", device=device)
retrieval_threshold = 28
retriever = ObjathorRetriever(
  clip_model=clip_model,
  clip_preprocess=clip_preprocess,
  clip_tokenizer=clip_tokenizer,
  sbert_model=sbert_model,
  retrieval_threshold=retrieval_threshold,
  device=device
)
logger.info('load models complete')

database = retriever.database
def get_annotations(obj_data):
  if "annotations" in obj_data:
      return obj_data["annotations"]
  else:
      # The assert here is just double-checking that a field that should exist does.
      assert "onFloor" in obj_data, f"Can not find annotations in obj_data {obj_data}"

      return obj_data

def softmax_sampling(candidates, temperature=1.0):
  scores = np.array([score for _, score in candidates], dtype=np.float64)
  
  # 使用温度调整得分，然后计算softmax概率
  adjusted_scores = scores / temperature
  exp_scores = np.exp(adjusted_scores)
  probabilities = exp_scores / np.sum(exp_scores)
  
  # 根据概率进行随机采样，返回对应的候选人
  sampled_index = np.random.choice(len(candidates), p=probabilities)
  sampled_candidate = candidates[sampled_index]
  
  return sampled_candidate

  
def retrieve_furniture(scene_dir, sample, temperature):
  furniture_list = []
  path = os.path.join(scene_dir, '10_full_scene_graph.json')
  try:
    furniture_layout = json.load(
      open(path)
    )
  except FileNotFoundError:
    logger.error(f'File not found: {path}')

  room_dimension = furniture_layout['room_dimension']
  for area_idx, area in enumerate(tqdm(furniture_layout['functional_area'], desc='Floor Furniture', leave=False)):
    for fur_idx, fur in enumerate(area['furnitures']):
      name = fur['name']
      description = [f"{name}, {fur['description']}"]
      candidates = retriever.retrieve(description, threshold=10)#, target_size=fur['size'])
      candidates = [
        candidate
        for candidate, annotation in zip(
            candidates,
            [
                get_annotations(database[candidate[0]])
                for candidate in candidates
            ],
        )
        if annotation["onFloor"]  # only select objects on the floor
        and (
            not annotation["onCeiling"]
        )  # only select objects not on the ceiling
        and all(  # ignore doors and windows and frames
            k not in annotation["category"].lower()
            for k in ["door", "window", "frame"]
        )
      ]
      try:
        candidates = retriever.compute_size_difference(
          [s / 100 for s in fur['size']], candidates
        )
      except Exception as e:
        logger.error(f'{name}, {candidates}, {description}')
        raise e

      try:
        if sample == 'top':
          choose = candidates[0]
        else:
          choose = softmax_sampling(candidates, temperature=temperature)

        if True:
          for c in candidates[:1]:
            logger.debug(description[0])
            logger.debug(get_annotations(database[c[0]]))
            logger.debug(get_bbox_dims(database[c[0]]))
            logger.debug(fur['size'])
        uid = choose[0]
        furniture_layout['functional_area'][area_idx]['furnitures'][fur_idx]['uid'] = uid
        furniture_layout['functional_area'][area_idx]['furnitures'][fur_idx]['supported'] = 'floor'
        # furniture_layout['functional_area'][area_idx]['furnitures'][fur_idx]['location'].append(0)
        try:
          furniture_layout['functional_area'][area_idx]['furnitures'][fur_idx].pop('placement_rule')
        except:
          pass
        try:
          furniture_layout['functional_area'][area_idx]['furnitures'][fur_idx].pop('color')
        except:
          pass
        try:
          furniture_layout['functional_area'][area_idx]['furnitures'][fur_idx].pop('anchor')
        except:
          pass

        furniture_list.append({
          name: furniture_layout['functional_area'][area_idx]['furnitures'][fur_idx]
        })
      except Exception as e:
        logger.error(f'{name}, {candidates}, {description}')
        raise e

  small_object_list = []

  for supported in tqdm(furniture_layout["small_objects"], desc='Small Object', leave=False):
    fur_name = supported['name']
    fur = None
    for _ in furniture_list:
      if list(_.keys())[0] == fur_name:
        fur = _[list(_.keys())[0]]
        break
    assert fur is not None
    # fur_height = fur['location'][2] + fur['size'][2]

    for obj in tqdm(supported['small_objects'], leave=False):
      description = obj['name']
      obj_name = obj['name']
      candidates = retriever.retrieve(description, threshold=20)#, target_size=obj['size'])
      candidates = [
        candidate
        for candidate, annotation in zip(
            candidates,
            [
                get_annotations(database[candidate[0]])
                for candidate in candidates
            ],
        )
        if annotation["onFloor"]  # only select objects on the floor
        and (
            not annotation["onCeiling"]
        )  # only select objects not on the ceiling
        and all(  # ignore doors and windows and frames
            k not in annotation["category"].lower()
            for k in ["door", "window", "frame"]
        )
      ]
      try:
        if sample == 'top':
          choose = candidates[0]
        else:
          choose = softmax_sampling(candidates, temperature=temperature)
        uid = choose[0]
        obj['uid'] = uid
        obj['supported'] = fur_name
        # obj['location'].append(fur_height)
        try:
          obj.pop('anchor')
        except:
          pass
        try:
          obj.pop('placement_rule')
        except:
          pass
        try:
          obj.pop('color')
        except:
          pass

        small_object_list.append({
          f"{fur_name}_{obj_name}": obj
        })
      except Exception as e:
        logger.error(f'{name}, {candidates}, {description}')
        raise e

    result = {
      'room_dimension': room_dimension,
      'objects': furniture_list + small_object_list
    }

    json.dump(result, open(os.path.join(scene_dir, '11_retrieved_results.json'), 'w'), indent=2)  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument( '--project_dir')
  parser.add_argument( '--sample', type=str, choices=['top', 'prob'])
  parser.add_argument( '--temperature', type=float, default=0.7)
  args = parser.parse_args()
  path_list = os.listdir(args.project_dir)
  path_list.sort(key = lambda x: int(x.split('_')[0]))
  logger.info(path_list)
  for scene_dir in tqdm(path_list[:]):
    logger.info(f'processing {scene_dir}')
    retrieve_furniture(os.path.join(args.project_dir, scene_dir), args.sample, args.temperature)
