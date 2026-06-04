from pydantic import BaseModel, Field, field_validator
from typing import List
from collections import Counter

class RegionLayout(BaseModel):
  region_name: str = Field(description="The name of the region")
  region_room_relation: int = Field(description="The index of the relation between room and region")
  region_length: float = Field(description="The length of the region", gt=0.5, lt=8.0)

def get_response_mode(num_region, layout_names):
  # RegionLayout = get_region_layout_model(choices)
  class RegionLayoutResponseModel(BaseModel):
    output: List[RegionLayout] = Field(description="The layout of the regions", min_items=num_region, max_items=num_region)

    @field_validator('output')
    def check(response):
      cnt = Counter({name: 0 for name in layout_names})
      for idx, area in enumerate(response):
        if area.region_name not in layout_names:
          raise ValueError(f"The region name {area.region_name} is not in the layout names {layout_names}")
        cnt[area.region_name] += 1
      if len(cnt) != num_region:
        raise ValueError(f"The number of regions is not equal to the number of the input regions. You output {len(cnt)} regions but required {num_region} regions: {cnt}")
      for name, cnt in cnt.items():
        if cnt != 1:
          raise ValueError(f"The region name {name} is output {cnt} times but required 1 time")
      return response
  return RegionLayoutResponseModel


def fmt(areas, anchor_region, room_desc, room_dimension, num_region_2_relation):
  ret = {}
  ret['user_prompt'] = \
f"""
You are an expert interior designer tasked with analyzing a room layout. Based on the provided room type, size, description, existing functional areas and corresponding descriptions, your job is to determine the layout of the functional regions in the room.
Input:
- Room Description: {room_desc}
- Functional Regions ({areas[2]} regions):
{areas[0]}
- Main Region: {anchor_region}

You need to determine the layout of the functional regions. The following options describe the relation of the regions to the room. You need to specify distinct a index (an integer) for each region:
{num_region_2_relation[areas[2]]}

Then, you need to determine a reasonalbe length of each region in meters. 
You should consider the length of the entire room.
You should make the length of the main region (i.e. {anchor_region}) not shorter than the others.
You should not make the regions too short or too long.
Make sure sum of the lengths of each region is equal to the length of the room. (i.e. l_1 + l_2 + ... + l_3 = L_total)
Here is an example:
---
Input:
- Room Type: living room
- Room Description: a warm and inviting living room
- Room Size: 
  - large: length = X meters
- Functional Regions:
  - Rest Area: The main area for relaxation, watching TV, chatting, or unwinding
  - Dining Area: For daily meals or entertaining guests
  - Storage Area: Storing everyday items, books, electronic devices, or decorations, keeping the living room organized

Output:
  - Rest Area: right side, length = x_1 (meters)
  - Dining Area: center, length = x_2 (meters)
  - Storage Area: left side, length  = x_3 (meters)
---
"""
      
  ret['check_fn'] = None
  return ret

def get_region_layout(areas, anchor_region, room_desc, room_dimension, get_response_task):
  num_region_2_relation = {
    1: \
"""  - [1]: center
""",
    2: \
"""  - [1]: left side
  - [2]: right side
""",
    3: \
"""  - [1]: left side
  - [2]: center
  - [3]: right side
""",
    4: \
"""  - [1]: left side
  - [2]: left center
  - [3]: right center
  - [4]: right side
""",
    5: \
"""  - [1]: left side
  - [2]: left center
  - [3]: center
  - [4]: right center
  - [5]: right side
"""
  }
  response = get_response_task(task='REGION ATTR', **fmt(areas, anchor_region, room_desc, room_dimension, num_region_2_relation), response_model=get_response_mode(areas[2], areas[1]))
  for idx, area in enumerate(response['output']):
    response['output'][idx]['region_dimension'] = [area['region_length'], room_dimension[1]]
  return response
