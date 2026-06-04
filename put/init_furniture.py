from utils.extract_json import extract_json_from_response
import json
from random import choice, uniform
import numpy as np

def softmax(x):
  x -= np.max(x)
  x = np.exp(x) / np.sum(np.exp(x))
  return x


def fmt(obj_name, obj_desc, placement_policy_desc, obj_dimension, floor_np):
  ret = {}
  ret['user_prompt'] = \
f"""
[Role]
You are a professional indoor designer. 

[Task]
Your task is to determine the exact cell indices from the input ground (floor) where the input furniture should be placed.

[Input]
Furniture Name: {obj_name}
Furniture Description: {obj_desc}
Furniture Size: the furniture will occupy {obj_dimension[0]} rows and {obj_dimension[1]} columns
Furniture Placement Rule: {placement_policy_desc}
Ground (Floor): The ground (floor) is the input image, which is segmented as a grid and each cell is marked with a number. Around the grid, there are walls or boundaries. Furnitures should not be placed on walls or boundaries.
  - Note that: if a furniture is required to be placed in the middle of a region, it can by placed near the boundaries and in the middle of the grid but cannot be placed near the walls. In contrast, if a furniture is required to be placed near a wall, it cannot be placed near a boundary or in the middle of the grid.

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input, first assess the situation and explain how you are approaching the task.
2. **Break Down the Problem**: Identify key components of the task and explain the reasoning behind each step.
3. **Step-by-step Execution**: For each step, describe what you're doing, and why, providing reasoning behind decisions made. Ensure that intermediate outputs are included at each stage.
4. **JSON Outputs**: After you output the intermediate results. You need to output a JSON.

[Task Requirement]
1. You need to determine the cells where the furniture shoule be placed. Please consider the placement rule in the input.
2. The number of the output cells must equals to rows times columns of the furniture.

[JSON Format]
Organize the results in the following JSON format (make sure the JSON is surrounded by a pair of ```json and ```, do not include "//" comments):
```json
{{
  "output": {{
    "number of cells": "<Based on the input size of the furniture, how many cells should the furniture occupied>",
    "{min(obj_dimension)} column(s) near a wall": "<From the input image, output {min(obj_dimension)} column(s) that near a wall>",
    "where to put the furniture": "<Based on the input furniture placement rule, answering where should the furniture be placed. Answering the exact indices of the cells that satisfy the constraint>",
    "indices": [index-1, index-2, ..., index-N]
  }}
}}
```
"""

  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```, no '//' comment")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
      
  ret['check_fn'] = check
  ret['image_np'] = floor_np
  return ret


def place_wall(wall, bound, obj_size, front_side, threshold=0.1):
  eps = 1e-3
  candidate = list(wall).copy()
  room_x = bound[1] - bound[0]
  room_y = bound[3] - bound[2]
  x_min, x_max, y_min, y_max = bound
  prob = [room_x, room_y, room_x, room_y]

  if not (obj_size[0] < room_x + eps and obj_size[1] < room_y + eps):
    candidate[0] = 0
    candidate[2] = 0
  if not (obj_size[1] < room_x + eps and obj_size[0] < room_y + eps):
    candidate[1] = 0
    candidate[3] = 0

  if sum(candidate) <= 0:
    return False, None

  candidate = [idx for idx, value in enumerate(candidate) if value == 1]
  prob = np.array([prob[idx] for idx in candidate])
  prob = softmax(prob * 3.0)
  side = np.random.choice(a=np.array(candidate), p=prob)
  orientation = [2, 3, 0, 1]


  if side in [0, 2]:
    if side == 0:
      center_y = y_max - obj_size[1] / 2
    else:
      center_y = y_min + obj_size[1] / 2
    
    center_x = uniform(x_min + obj_size[0] / 2, x_max - obj_size[0] / 2)
    if x_min + obj_size[0] / 2 + threshold >= center_x:
      center_x = x_min + obj_size[0] / 2
    if center_x >= x_max - obj_size[0] / 2 - threshold:
      center_x = x_max - obj_size[0] / 2
  else:
    obj_size[0], obj_size[1] = obj_size[1], obj_size[0]

    if side == 1:
      center_x = x_max - obj_size[0] / 2
    else:
      center_x = x_min + obj_size[0] / 2
    
    center_y = uniform(y_min + obj_size[1] / 2, y_max - obj_size[1] / 2)
    if y_min + obj_size[1] / 2 + threshold >= center_y:
      center_y = y_min + obj_size[1] / 2
    if center_y >= y_max - obj_size[1] / 2 - threshold:
      center_y = y_max - obj_size[1] / 2

  return True, {
    'output': {
      'side': side,
      'center_x': center_x,
      'center_y': center_y,
      'size': obj_size,
      'orientation': orientation[side],
      "candidates": candidate
    },
  }


def place_center(wall, bound, obj_size, front_side, threshold=0.1):
  eps = 1e-3
  orientation_candidate = [1, 1, 1, 1]
  room_x = bound[1] - bound[0]
  room_y = bound[3] - bound[2]
  x_min, x_max, y_min, y_max = bound

  if not (obj_size[0] < room_x + eps and obj_size[1] < room_y + eps):
    orientation_candidate[0] = 0
    orientation_candidate[2] = 0
  if not (obj_size[1] < room_x + eps and obj_size[0] < room_y + eps):
    orientation_candidate[1] = 0
    orientation_candidate[3] = 0

  if sum(orientation_candidate) == 0:
    return False, None

  orientation_candidate = [idx for idx, value in enumerate(orientation_candidate) if value == 1]
  orientation = choice(orientation_candidate)
  
  center_x = (x_min + x_max) / 2
  center_y = (y_min + y_max) / 2

  if orientation in [0, 2]:
    obj_size[0], obj_size[1] = obj_size[0], obj_size[1]
  else:
    obj_size[0], obj_size[1] = obj_size[1], obj_size[0]

  return True, {
    'output': {
      'center_x': center_x,
      'center_y': center_y,
      'size': obj_size,
      'orientation': orientation,
      'side': 'center'
    },
  }


def place_corner(wall, bound, obj_size, front_side, threshold=0.1):
  ...