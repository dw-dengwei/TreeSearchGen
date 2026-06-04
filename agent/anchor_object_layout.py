from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field

class AnchorPlacementResponseModel(BaseModel):
  placement: str = Field(description="The placement of the anchor object", choices=['place_center', 'place_wall', 'place_corner'])


def fmt(area_name, area_desc, area_dimension,obj_name, obj_desc):
  ret = {}
  ret['user_prompt'] = \
f"""
You are an expert interior designer tasked with analyzing a room layout. Based on the provided room type, description, a functional region and corresponding description, and an anchor furniture which represents the main function of the input functional region. Your job is to determine the size and placement policy for the input anchor furniture.
Input:
- Functional Region: {area_name}
- Functional Region Description: {area_desc}
- Functional Region Size: length = {area_dimension[0]} meters, width = {area_dimension[1]} meters
- Anchor Furniture: {obj_name}
- Anchor Furniture Description: {obj_desc}

You should decide on the placement rule for this anchor furniture. You can use only the following anchor rules:
(1) "place_center" which places the anchor furniture at the center of the room.
(2) "place_wall" which places the anchor with its back against a segment of the wall.
(3) "place_corner" which places the anchor at a corner.

You need to format your output in a valid JSON (ensure the json is surrounded by ```json and ```, no "//" comment):
"""
  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```, no '//' comment")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
    if js['output']['placement'] not in ['place_center', 'place_wall', 'place_corner']:
      raise Exception(f"Your last output has errors. Please fix it in this term: The placement rule is not one of 'place_center', 'place_wall', 'place_corner':\n{response}\n\n")
      
  ret['check_fn'] = check
  return ret

def get_anchor_placement_attr(area_name, area_desc, area_dimension, obj_name, obj_desc, get_response_task):
  response = get_response_task(task='ANCHOR OBJ LAYOUT', **fmt(area_name, area_desc, area_dimension, obj_name, obj_desc), response_model=AnchorPlacementResponseModel)
  response = {"output": response}
  return response