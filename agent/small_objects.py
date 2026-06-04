from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

class Size(BaseModel):
  length: float = Field(description="The length of the object")
  width: float = Field(description="The width of the object")
  height: float = Field(description="The height of the object")

class SmallObject(BaseModel):
  name: str = Field(description="The name of the small object")
  size: Size = Field(description="The size of the small object")
  description: str = Field(description="The description of the small object")
  placement_rule: str = Field(description="The placement rule of the small object. place_center and place_back_side_center for anchor small objects. place_front and place_around for non-anchor small objects.")
  frontal: str = Field(description="The frontal side of the small object", choices=['longer', 'shorter'])

class FurnitureItem(BaseModel):
  name: str = Field(description="The name of the furniture")
  is_table_like: str = Field(description="Whether the furniture is table like", choices=['yes', 'no'])
  small_objects: List[SmallObject] = Field(description="The small objects that should be placed on the furniture")
  anchor_small_object: Optional[str] = Field(description="Name of the anchor small object. None if the furniture is not table like")
  
  @model_validator(mode='after')
  def validate_placement_rules(self):
    """验证anchor物体和非anchor物体的placement rule是否符合要求"""
    small_objects = self.small_objects
    anchor_name = self.anchor_small_object
    
    if not small_objects or anchor_name is None:
      return self
      
    for obj in small_objects:
      # 如果是anchor物体
      if obj.name == anchor_name:
        if obj.placement_rule not in ['place_center', 'place_back_side_center']:
          raise ValueError(
            f"Anchor object '{obj.name}' must have placement_rule 'place_center' or 'place_back_side_center', got '{obj.placement_rule}'"
          )
      # 如果不是anchor物体
      else:
        if obj.placement_rule not in ['place_front', 'place_around']:
          raise ValueError(
            f"Non-anchor object '{obj.name}' must have placement_rule 'place_front' or 'place_around', got '{obj.placement_rule}'"
          )
    
    return self

class SmallObjectsResponseModel(BaseModel):
  output: List[FurnitureItem] = Field(description="List of furniture items with their small objects")


def fmt(furnitures):
  ret = {}
  ret['user_prompt'] = \
f"""
[Role]
You are a professional indoor designer.

[Task]
Your task is to determine small objects for each furnitures.

[Input]
{furnitures}

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input, first assess the situation and explain how you are approaching the task.
2. **Break Down the Problem**: Identify key components of the task and explain the reasoning behind each step.
3. **Step-by-step Execution**: For each step, describe what you're doing, and why, providing reasoning behind decisions made. Ensure that intermediate outputs are included at each stage.
4. **JSON Outputs**: After you output the intermediate results. You need to output a JSON.

[Task Requirement]
The input is the furnitures and corresponding information in a room. You follow the below steps:
1. Determine if the furniture is table-like or not. For example:
  - chair: no
  - bed: no
  - sofa: no
  - dresser: no
  - table: yes
  - TV stand: yes
  - nightstand: yes
  - desk: yes
  - dining table: yes
2. For the furnitures that are table-like, generate a list of small objects and corresponding descriptions (describe the appearance, function, and usage of each small object) that can be placed on it. The small objects should be typically placed on the furniture for functional purposes. Besides, you need to provide a reasonable size (length, width, height in meters) for each small objects and determine an anchor object among the small objects for each table-like furniture. An anchor object is typically the largest object among the objects placed on the same furniture.
3. For the furnitures that are not table-like, just provide an empty list.
4. For each **anchor small object**, determine the placement rules from the following options:
  - place_center: place the small object in the center of the supported furniture. For example, place a vase on the center of the dining table.
  - place_back_side_center: place the small object in the center of the back side of the supported furniture and remains a free space in front of the small object. For example, place a monitor on the back side center of the office table. It remains a free space in front of the monitor for users to place a keyboard and other objects.
5. For the small objects that are **not anchor objects**, determine the placement rules. The placement rules are relevant to the anchor small object. You need to choose from the following options:
  - place_front: place the small object in front of the anchor small object. For example, place a keyboard in front of the monitor on the office table.
  - place_around: place the small object around the anchor small object. For example, place a notebook around the monitor on the office table.
6. You need to determine the head of this furniture in the top-down view from the following options:
  - longer: means the object's head (frontal side) is commonly the longer side in the top-down view rather than the shorter side in top-down view for functional usage. For example: 
    - monitor
    - tv
    - alarm clock
  - shorter: means the object's head (frontal side) is commonly the shorter side in the top-down view rather than the longer side in top-down view for functional usage. For example: 
    - photo frame
    - book
  (If the object is not in the examples, you can choose 'longer' for this task.)

7. You should pay attention that: the placement rules for the anchor small objects should be chosen from "place_center" and "place_back_side_center". 
8. The placement rules for the small objects that are not anchor objects should be chosen from "place_front" and "place_around".

"""

  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```, no '//' comment")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
      
  ret['check_fn'] = check
  return ret

def get_small_objects(input_info, get_response_task):
  response = get_response_task(task='SIZE', **fmt(input_info), response_model=SmallObjectsResponseModel)
  for fur in response['output']:
    for small_obj in fur['small_objects']:
      small_obj['size'] = [small_obj['size']['length'], small_obj['size']['width'], small_obj['size']['height']]
  
  return response

