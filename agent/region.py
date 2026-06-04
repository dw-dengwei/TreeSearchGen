from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field
from typing import List, Dict

class Region(BaseModel):
  name: str = Field(description="The name of the region")
  description: str = Field(description="The description of the region")
  relationship: Dict[str, str] = Field(description="The relationship of the region to other regions")

class RegionResponseModel(BaseModel):
  room_desc: str = Field(description="The description of the room")
  functional_zones: List[Region] = Field(description="The regions that the furniture should be placed", min_items=1, max_items=5)


def fmt(room_desc):
  ret = {}
  ret['user_prompt'] = \
f"""
You are an intelligent assistant tasked with categorizing functional areas in a room based on its type, description, and size. Your goal is to carefully analyze the provided information and determine which functional areas from the list are suitable for the given room. Follow these steps to ensure accurate results:

1. **Analyze Room Type**: Identify whether the room is a "living room", "bedroom.", "kitchen", or "bathroom".
2. **Evaluate Room Description**: Read the detailed description of the room to understand its features, layout, and size.
3. **Determine Room Size**: Assess the size of the room (e.g., small, medium, large) based on the description.
4. **Select Functional Areas**: From the provided functional areas, choose one or more that best fit the room type and size. Ensure that:
   - For small rooms, prioritize practicality and consider limiting to one or two functional areas.
   - For medium and large rooms, you can select multiple functional areas if they can be accommodated comfortably.
5. **Describe Spatial Relationships**: For each selected functional area, explain its position relative to others (e.g., to the left, behind, adjacent).

Here are some example functional areas. You can choose from them but not limited to them:
- Rest Area
- Dining Area
- Storage Area
- Work Area
- Cooking Area
- Bathing Area
- etc.

The relationships you can choose from are:
- left of
- right of
- behind
- in front of

Input:
1. Room Description: {room_desc}
"""

  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
    if 'functional_zones' not in js['output'].keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `functional_zones` key:\n{response}\n\n")
      
  ret['check_fn'] = check
  return ret

def get_regions(description, get_response_task):
  response = get_response_task(task='REGION', **fmt(description), response_model=RegionResponseModel)
  response = {"output": response}
  return response
