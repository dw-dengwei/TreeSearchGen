from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field
from typing import List

class Appearance(BaseModel):
  name: str = Field(description="The name of the furniture")
  appearance: str = Field(description="The appearance description for the furniture", max_length=25)

def get_response_model(object_count):
  class AppearanceResponseModel(BaseModel):
    reasoning: str = Field(description="The reasoning for the appearance description")
    furniture_list: List[Appearance] = Field(description="The appearance description for each furniture", max_items=object_count, min_items=object_count)
    floor_texture: str = Field(description="The texture description for the floor", max_length=25)
  
  return AppearanceResponseModel


def fmt(objects, room_desc):
  ret = {}
  ret['user_prompt'] = \
f"""
Generate appearance descriptions for a furniture based on the following:
Furniture List: 
{objects}
Room Description:
{room_desc}

[Requirements]
1. Focus only on appearance aspects: color, texture, material finish, decorative details.
2. Do not describe geometry, shape, or structure.
3. Ensure all items follow a unified and coherent appearance philosophy.
4. Use concise phrases (max 25 characters) to describe each item's appearance.
5. Each description should focus on 1-2 key visual features only.
6. For floor texture, use a similarly concise description.
7. Use professional terminology.
8. You should output your reasoning before the final answer.
9. The appearance description should be consistent with the global room description.
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

def get_appearance_desc(objects, room_desc, get_response_task):
  num_objects = len(objects)
  objects = '\n - '.join(objects)
  objects = " - " + objects
  response = get_response_task(task='appearance', **fmt(objects, room_desc), response_model=get_response_model(num_objects))
  response = {"output": response}
  return response