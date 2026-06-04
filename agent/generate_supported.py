from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field
from typing import List

class Size(BaseModel):
  length: float = Field(description="The length of the object")
  width: float = Field(description="The width of the object")
  height: float = Field(description="The height of the object")

class Furniture(BaseModel):
  name: str = Field(description="The name of the furniture")
  description: str = Field(description="The description of the furniture")
  size: Size = Field(description="The size of the furniture")
  placement: str = Field(description="Where should the furniture be placed, on floor or on other furnitures", choices=['floor', 'other'])
  frontal: str = Field(description="The frontal side of the furniture", choices=['longer', 'shorter'])
  reason: str = Field(description="The reason for the decision")

class Area(BaseModel):
  name: str = Field(description="The name of the area")
  description: str = Field(description="The description of the area")

class SupportedResponseModel(BaseModel):
  area: Area = Field(description="The area that the furniture should be placed in")
  furnitures: List[Furniture] = Field(description="The furniture that should be placed in the area", min_items=2)


def fmt(area, size, get_response_check):
  ret = {}
  ret['user_prompt'] = \
f"""
Task:
You are an expert in interior design, and your task is to add furniture for a specific function area within a defined room type.

Input:
Functional Area: {area['name']}
Functional Area Description: {area['desc']}
Functional Area Size: {size[0]} meters length x {size[1]} meters width

Please follow these guidelines:
1. Do not provide rug/mat, windows, doors, curtains, floor, and ceiling objects which have been installed for each room.
2. You need to provide a description for each furniture. The description should include the type, function and appearance.
3. You need to estimate a reasonable size (in meters) for each furniture. For the size of each furniture, you should consider the size of the functional area. The size of the furniture should be reasonable and should not be too large.
4. Based on the function of the input furniture, determine the head of this furniture in the top-down view from the following options:
  - longer: means the furniture's head (frontal side) is commonly the longer side in the top-down view rather than the shorter side in top-down view for functional usage. For example: 
    - office table: In the top-down view, the longer side of an office table is used to provide working space for users. Thus, the longer side of an office table is commonly regarded as the frontal side.
    - nightstand: In the top-down view, the longer side of a nightstand is used to provide storage space for users to put things. Thus, the longer side of a nightstand is commonly regarded as the frontal side.
  - shorter: means the furniture's head (frontal side) is commonly the shorter side in the top-down view rather than the longer side in top-down view for functional usage. For example: 
    - bed: In the top-down view, the shorter side of a bed is used to place pillows and support head and neck. Thus, the shorter side is typically regarded as the head (frontal side) of a bed.
    - bathtub: In the top-down view, the shorter side of a bathtub is used to place the head of users. Thus, the shorter side of a bathtub is typically regarded as the head (frontal side).
    - toilet: In the top-down view, users sit on the shorter side of a toilet. Thus, the shorter side of a toilet is typically regarded as the head (frontal side).
  (If the furniture is not in the examples, you can choose 'longer' for this task.)
5. You must provide main furniture for the area. The main furniture is the one that is typically used in the area and it can present the most important features of the area. For example:
  - a bed for the resting area of a bedroom
6. Do not use plural words in your response. Use singular form for all words (e.g., use "chair" instead of "chairs").
7. The furniture must typically in this specific functional region of the room.
8. The furniture must typically be placed directly on the floor or ground. The object that is usually placed on other furniture is not allowed.
  - item that is typically placed on the floor. This item should not be supported by other furniture. Examples include:
    - sofa
    - bed
    - etc.
  - item that is placed on other furniture. Examples include:
    - television (on television stand)
    - desk lamp (on desk)
    - etc.
9. You must consider the size of the area. The number of furniture you generated should not be too many that the area cannot fit comfortably.
10. If the area has enough space, you can should provide as much furniture as possible.
"""

  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
    objects = []
    for obj in js['output']['furnitures']:
      objects.append(
        f"""- {obj['name']}: {obj['description']}"""
      )
    if len(objects) < 2:
      raise Exception(f"Your last output has errors. Please fix it in this term: You must provide at least 2 furnitures:\n{response}\n\n")
    for fur in js['output']['furnitures']:
      if fur['frontal'].lower() not in ['longer', 'shorter']:
        raise Exception(f"Your last output has errors. Please fix it in this term: The `frontal` field should be either 'longer' or 'shorter':\n{response}\n\n")
      
  ret['check_fn'] = check
  return ret

def get_supported(area, size, get_response_task, get_response_check):
  response = get_response_task(task='OBJ', **fmt(area, size, get_response_check), response_model=SupportedResponseModel)
  for fur in response['furnitures']:
    fur['size'] = [fur['size']['length'], fur['size']['width'], fur['size']['height']]
  response = {"output": response}
  return response