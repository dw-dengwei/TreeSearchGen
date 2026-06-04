from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field, field_validator

class OtherPlacementResponseModel(BaseModel):
  reasoning: str = Field(description="The reasoning for the decision")
  placement_rule: str = Field(description="The placement rule of the affiliated furniture", choices=['place_front', 'place_beside', 'place_around'])
  functionally_grouped: str = Field(description="The functionally grouped of the affiliated furniture", choices=['yes', 'no'])
  close_to_the_wall: str = Field(description="The close to the wall of the affiliated furniture", choices=['yes', 'no'])
  distance: str = Field(description="The distance of the affiliated furniture", choices=['adjacent_to', 'near', 'far'])
  alignment: str = Field(description="The alignment of the affiliated furniture", choices=['side alignment', 'center alignment', 'around'])

  @field_validator('placement_rule')
  @classmethod
  def validate_placement_rule(cls, v: str, info) -> str:
    if v not in ['place_front', 'place_beside', 'place_around']:
      raise ValueError(f"Invalid placement rule: {v}. Must be one of ['place_front', 'place_beside', 'place_around']")
    return v
  
  @field_validator('functionally_grouped')
  @classmethod
  def validate_functionally_grouped(cls, v: str, info) -> str:
    if v not in ['yes', 'no']:
      raise ValueError(f"Invalid functionally grouped: {v}. Must be one of ['yes', 'no']")
    return v
  
  @field_validator('close_to_the_wall')
  @classmethod
  def validate_close_to_the_wall(cls, v: str, info) -> str:
    if v not in ['yes', 'no']:
      raise ValueError(f"Invalid close to the wall: {v}. Must be one of ['yes', 'no']")
    return v
  
  @field_validator('distance')
  @classmethod
  def validate_distance(cls, v: str, info) -> str:
    if v not in ['adjacent_to', 'near', 'far']:
      raise ValueError(f"Invalid distance: {v}. Must be one of ['adjacent_to', 'near', 'far']")
    return v
  
  @field_validator('alignment')
  @classmethod
  def validate_alignment(cls, v: str, info) -> str:
    if v not in ['side alignment', 'center alignment', 'around']:
      raise ValueError(f"Invalid alignment: {v}. Must be one of ['side alignment', 'center alignment', 'around']")
    return v

def fmt(area_name, area_desc, anchor_name, anchor_desc, obj_name, obj_desc, room_desc):
  ret = {}
  ret['user_prompt'] = \
f"""
You are an expert interior designer tasked with analyzing a room layout. Based on the provided room type, description, a functional region and corresponding description, an anchor furniture which represents the main function of the input functional region, and some affiliated furnitures which presents the secondary function of the region. Your job is to determine the size and placement policy for the affiliated furnitures.
Input:
- Functional Region: {area_name}
- Functional Region Description: {area_desc}
- Anchor Furniture: {anchor_name}
- Anchor Furniture Description: {anchor_desc}
- Affiliated Furniture: {obj_name}  
- Affiliated Furniture Description: {obj_desc}

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input, first assess the situation and explain how you are approaching the task.
2. **Break Down the Problem**: Identify key components of the task and explain the reasoning behind each step.
3. **Step-by-step Execution**: For each step, describe what you're doing, and why, providing reasoning behind decisions made. Ensure that intermediate outputs are included at each stage.

Step 1: You need to determine the placement policy for each affiliated furniture. A placement policy means where should the affiated furniture be placed relative to the anchor furniture ({anchor_name}). You must consider the common sences of the indoor layout design. You should consider the functionality of the furniture. The answer must be chosen from the following options:
(1) "place_front" which places the furniture in front of the anchor funiture ({anchor_name}). For example:
  - TV stand in front of a sofa.
  - Office chair in front of a desk.
  - Coffee table in front of a sofa.
(2) "place_beside" which places the furniture beside the anchor funiture ({anchor_name}). For example:
  - Nightstand beside a bed.
(3) "place_around" which places the furniture around the anchor funiture ({anchor_name}). For example:
  - chair around a dining table.

Step 2: You need to determine if the affiliated furniture and the anchor furniture are highly functionally grouped together. That means users usually put them together as well as use them together.
For example:
  - Bed and Nightstand: The bed and nightstand are functionally grouped together for users to sleep in the bed and reach the nightstand easily.
  - Table/Desk and Chair: The table/desk and chair are functionally grouped together for users to sit down on the chairs and work or eat on the desk.
  - Sofa and Coffee Table: The sofa and coffee table are functionally grouped together for users to relax on the sofa and read a book or drink coffee on the coffee table.
  - Sofa and TV stand: The sofa and TV stand are functionally grouped together for user to sit on the sofa and watch TV.
Most pair of furnitures are not functionally grouped together. If the input furnitures are not present in the "functionally grouped" examples, you should choose "no".

Step 3: You need to determine if the affiliated furniture should be placed close to the wall or in the middle of the room. For example:
  - Close to the wall:
    - Sofa
    - Bed
    - Nightstand
    - Wardrobe
    - Dresser
    - Closet
  - Not close to the wall:
    - coffee table
    - armchair

Step 4: You need to describe the distance of the two furnitures when putting them together. You should choose from the following options:
- "adjacent_to": means the two furnitures are placed close to (adjacent to) each other. For example:
  - nightstand and bed. The nightstand is usually placed adjacent to the bed for users to relax in the bed and easily reach the nightstand.
  - desk/table and chair. The chair is usually placed adjacent to the desk/table for users to sit and work.
- "near": means the two furnitures are placed near each other with a space between them. They are not close to (not adjacent to) each other. For example:
  - sofa and coffee table. The coffee table is usually placed near the sofa for users to relax on the sofa and read a book or drink coffee on the coffee table. It needs a space between them to walk around.
- "far": means the two furnitures are placed far from each other. For example:
  - sofa and TV stand. The TV stand is usually placed far from the sofa for comfortable viewing distance.

Step 5: You need to determine the alignment principle of the two furnitures. You need to choose from the following options:
- side alignment: the two furnitures are aligned by their sides. For example:
  - bed and nightstand: the side of the bed is aligned with the side of the nightstand for users to easily access the nightstand when they are in bed.
- center alignment: the two furnitures are aligned by their center. For example:
  - sofa and coffee table: the center of the sofa is aligned with the center of the coffee table for users to easily access the coffee table when they are sitting on the sofa.
  - sofa and tv stand: the center of the sofa is aligned with the center of the tv stand for better viewing experience.
  - desk and office chair/swivel chair: the center of the desk is aligned with the center of the chair for users to easily access the chair when they are working or studying.
- around: the affliated furniture is around the anchor furniture. For example:
  - dining table and chairs: the chairs are around the dining table for users to sit around the table.

You should first output the intermediate results. Then you need to format your output in an invalid JSON. The JSON should have the following format:
"""


  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
    fur = js['output']
    if fur['placement_rule'] not in ['place_front', 'place_beside', 'place_around']:
      raise Exception(f"Your last output has errors. Please fix it in this term: The placement_rule policy of the affiliated furnitures must be chosen from the following options: place_front, place_beside, place_around:\n{response}\n\n")
    if fur['functionally_grouped'] not in ['yes', 'no']:
      raise Exception(f"Your last output has errors. Please fix it in this term: The functionally_grouped must be chosen from the following options: yes, no:\n{response}\n\n")
    if fur['close_to_the_wall'] not in ['yes', 'no']:
      raise Exception(f"Your last output has errors. Please fix it in this term: The close_to_the_wall must be chosen from the following options: yes, no:\n{response}\n\n")
    if fur['distance'] not in ['adjacent_to', 'near', 'far']:
      raise Exception(f"Your last output has errors. Please fix it in this term: The distance policy of the affiliated furnitures must be chosen from the following options: adjacent_to, near, far:\n{response}\n\n")
    if fur['alignment'] not in ['side alignment', 'center alignment', 'around']:
      raise Exception(f"Your last output has errors. Please fix it in this term: The alignment policy of the affiliated furnitures must be chosen from the following options: side alignment, center alignment, around:\n{response}\n\n")

  ret['check_fn'] = check
  return ret

def get_other_placement_attr(area_name, area_desc, anchor_name, anchor_desc, obj_name, obj_desc, room_desc, get_response_task):
  response = get_response_task(task='OTHER OBJ LAYOUT', **fmt(area_name, area_desc, anchor_name, anchor_desc, obj_name, obj_desc, room_desc), response_model=OtherPlacementResponseModel)
  response = {"output": response}
  return response