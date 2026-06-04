from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field

def get_response_mode(areas):
  class AnchorRegionResponseModel(BaseModel):
    anchor_region_name: str = Field(description="The name of the anchor region", choices=areas)
  
  return AnchorRegionResponseModel


def fmt(room_desc, areas):
  ret = {}
  ret['user_prompt'] = \
f"""
You are tasked with classifying objects in a given room based on their role as either Anchor object or Other objects. An Anchor object is defined as a large item that represents the main function of a specified functional area. Other objects do not serve this primary role. Only 1 object can be classified as Anchor object.


Input:
- Room Description: {room_desc}
- Functional Areas:
{'\n'.join(list(map(lambda s: f"  -  {s['name']}: {s['desc']}", areas)))}

Please follow these steps:

1. Read the provided room type and description.
2. Understand the functional area and its description.
4. Select 1 Anchor functional based on its size and relevance to the main function of the room.
5. Create a JSON output that categorizes each object accordingly.

"""

  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
    anchor_name = js['output']['anchor-region-name']
    if anchor_name.lower().strip() not in list(map(lambda s: s['name'].lower().strip(), areas)):
      raise Exception(f"Your last output has errors. Please fix it in this term: the anchor region you selected is not in the input list. Make sure your output name of the region is exactly the same as the name in the input list (pay attention to case, spaces, dashes, etc.):\n{response}\n\n")
      
  ret['check_fn'] = check
  return ret

def get_anchor_region(room_desc, areas, get_response_task):
  response = get_response_task(task='GET ANCHOR REGION', **fmt(room_desc, areas), response_model=get_response_mode(list(map(lambda s: s['name'], areas))))
  response = {"output": response}
  return response

