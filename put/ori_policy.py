from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field
from enum import Enum


class OrientationRule(str, Enum):
  A = 'A'
  B = 'B'
  C = 'C'
  D = 'D'

class ResponseModel(BaseModel):
  reasoning: str = Field(description="The reasoning for the decision")
  orientation_rule: OrientationRule = Field(description="The orientation rule of the affiliated furniture")


def fmt(anchor_name, anchor_desc, obj_name, obj_desc):
  ret = {}
  ret['user_prompt'] = \
f"""
[Role]
You are a professional indoor designer. You know the placement rule of common furnitures.

[Task]
Your task is to determine the orientation of the affiliated furniture based on the function of the anchor object and the affiliated furniture.

[Input]
Anchor Furniture Name: {anchor_name}
Anchor Furniture Description: {anchor_desc}
Affiliated Furniture Name: {obj_name}
Affiliated Furniture Desc: {obj_desc}

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input, first assess the situation and explain how you are approaching the task.
2. **Break Down the Problem**: Identify key components of the task and explain the reasoning behind each step.
3. **Step-by-step Execution**: For each step, describe what you're doing, and why, providing reasoning behind decisions made. Ensure that intermediate outputs are included at each stage.
4. **JSON Outputs**: After you output the intermediate results. You need to output a JSON.

[Task Requirement]
You need to first understand the function of the anchor furniture and the affiliated furniture. Then, you need to choose the orientation rule for the affiliated furniture from the following options:
A. The affiliated furniture should face the position of the anchor furniture no matter which direction the anchor furniture faces.
B. The affiliated furniture should face back the position of the anchor furniture no matter which direction the anchor furniture faces.
C. The orientation of the affiliated furniture should depend on the direction of the anchor furniture. The affiliated furniture and the anchor furniture should face each other.
D. The orientation of the affiliated furniture should depend on the direction of the anchor furniture. The affiliated furniture and the anchor furniture should face to the same direction.

For example:
- **Sofa (anchor furniture) and TV stand (affiliaated furniture): The sofa and the TV stand should face each other so the user can comfortably watch TV. You should choose C because the sofa and TV stand must face each other for proper functionality.
- **Desk (anchor furniture) and office chair (affiliaated furniture): The office chair and the desk should face each other for the users to work comfortably. You should choose C because the desk and the office chair should face each other for practical use, regardless the position of the desk.
- **Dining table (anchor furniture) and dining chair (affiliaated furniture): Dining chair should face the dining table for proper seating during meals. You should choose A because the chairs should face towards the table for usability.
- **Bed (anchor furniture) and nightstand (affiliaated furniture): The nightstand should align with the bed in the same direction for a cohesive layout. You should choose D because the nightstand and bed should face the same direction to maintain order in the room.
"""

  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```, no '//' comment")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
    if js['output']['orientation_rule'] not in ['A', 'B', 'C', 'D']:
      raise Exception(f"Your last output has errors. Please fix it in this term: You must choose from A, B, C and D (one capital letter):\n{response}\n\n")
      
  ret['check_fn'] = check
  return ret

def get_ori_policy(anchor_name, anchor_desc, obj_name, obj_desc, get_response_task):
  response = get_response_task(task='ORI RULE', **fmt(anchor_name, anchor_desc, obj_name, obj_desc), response_model=ResponseModel)
  response = {"output": response}
  return response


