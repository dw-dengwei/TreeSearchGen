from pydantic import BaseModel, Field
from typing import List

def create_response_model(choices: List[str]):
    class ResponseModel(BaseModel):
        reasoning: str = Field(description="The reasoning for the decision")
        side: str = Field(description="The side of the anchor furniture that can accommodate the new furniture", choices=choices)
    
    return ResponseModel

def fmt(direction_options, direction_desc_candidate, axis, anchor_name, target_name, target_size, image_np):
  ret = {}
  ret['user_prompt'] = \
f"""
[Role]
You are a professional indoor designer. You know how to place furnitures in a room.

[Task]
Your task is to determine which side of the anchor furniture can accommodate the new furniture.

[Input]
Anchor Furniture Name: {anchor_name}, which is filled with red color and labled with its name.
Input Image: This image presents a floor of a room. The floor of the room is divided into a grid of cells. In this room, there is some furnitures. The furnitures are filled by red and labled with their names. In the free space, the cells are filled by emojis. Besides, the cells in the same {axis} share the same emoji. Around the floor, there are walls and boundaries. The walls are filled with the `wall` (which looks like a brick) emoji and the boundaries are filled with the `boundary` (which looks like a white circle) emoji. Both walls and boundaries cannot be occupied by furnitures. The walls and boundaries are different. The walls are the walls of a room and cannot be passed through. The boundaries are the boundaries of a region in the room and can be passed through.

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input, first assess the situation and explain how you are approaching the task.
2. **Break Down the Problem**: Identify key components of the task and explain the reasoning behind each step.
3. **Step-by-step Execution**: For each step, describe what you're doing, and why, providing reasoning behind decisions made. Ensure that intermediate outputs are included at each stage.
4. **JSON Outputs**: After you output the intermediate results. You need to output a JSON.

[Task Requirement]
1. I want to introduce a new furniture, "{target_name}", size = {target_size} {axis}s, into the room in the input image.
2. Please choose the side of the anchor furniture, "{anchor_name}" that is filled with red color and labled with its name, that can accommodate the new furniture. The side you choose should have enough {axis}s for the new furniture. The new furniture requires {target_size} {axis}s.
3. You are required to only choose from the following options and do not output other options:
  - {direction_options}
  (if both of them are suitable, you can choose either one.)
4. The cells that are filled with emojis except 'brick' and 'white circle' are available for the new furniture. The cells that are filled with 'brick', 'white circle' and are not filled with any emoji are not available for the new furniture.
5. You should first output the reasoning for the decision. Specifically, you should describe the input image in detail, including the layout, emojis, walls, boundaries, etc. Describe the emoji at the top, bottom, left, right of the anchor furniture. Describe the real color of the `bed`  
"""
      
  ret['check_fn'] = lambda _: None
  ret['image_np'] = image_np[:,:,::-1]
  return ret

def textual_fmt(direction_options, direction_desc_candidate, axis, anchor_name, target_name, target_size, text_layout):
  ret = {}
  ret['user_prompt'] = \
f"""
[Role]
You are a professional indoor designer. You know how to place furnitures in a room.

[Task]
Your task is to determine which side of the anchor furniture can accommodate the new furniture.

[Input]
Anchor Furniture Name: {anchor_name}, which is filled with red color and labled with its name.
Textual Layout: {text_layout}

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input, first assess the situation and explain how you are approaching the task.
2. **Break Down the Problem**: Identify key components of the task and explain the reasoning behind each step.
3. **Step-by-step Execution**: For each step, describe what you're doing, and why, providing reasoning behind decisions made. Ensure that intermediate outputs are included at each stage.
4. **JSON Outputs**: After you output the intermediate results. You need to output a JSON.

[Task Requirement]
1. I want to introduce a new furniture, "{target_name}", size = {target_size} {axis}s, into the room in the input image.
2. Please choose the side of the anchor furniture, "{anchor_name}" that is filled with red color and labled with its name, that can accommodate the new furniture. The side you choose should have enough {axis}s for the new furniture. The new furniture requires {target_size} {axis}s.
3. You are required to only choose from the following options and do not output other options:
  - {direction_options}
  (if both of them are suitable, you can choose either one.)
4. The cells that are filled with emojis except 'brick' and 'white circle' are available for the new furniture. The cells that are filled with 'brick', 'white circle' and are not filled with any emoji are not available for the new furniture.
5. You should first output the reasoning for the decision. Specifically, you should describe the input image in detail, including the layout, emojis, walls, boundaries, etc. Describe the emoji at the top, bottom, left, right of the anchor furniture. Describe the real color of the `bed`  
"""
      
  ret['check_fn'] = lambda _: None
  ret['image_np'] = None
  return ret


def get_offset_side(direction_desc_candidate, axis, anchor_name, target_name, target_size, image_np, text_layout, get_response_task, use_image=True):
    direction_opt = '\n'.join([f'  - {s}' for s in direction_desc_candidate])
    ResponseModel = create_response_model(direction_desc_candidate)

    if use_image:
      response = get_response_task(task='CHOOSE_SIDE', **fmt(direction_opt, direction_desc_candidate, axis, anchor_name, target_name, target_size, image_np), response_model=ResponseModel)
    else:
      response = get_response_task(task='CHOOSE_SIDE', **textual_fmt(direction_opt, direction_desc_candidate, axis, anchor_name, target_name, target_size, text_layout), response_model=ResponseModel)

    response = {"output": response}
    if use_image:
      response['output']['side'] = response['output']['side'].strip().lower()
    return response


if __name__ == '__main__':
    choices = ['left', 'right', 'top', 'bottom']
    ResponseModel = create_response_model(choices)
    response = ResponseModel(reasoning='', side='left')
    print(ResponseModel.model_json_schema())
    print(response.model_dump_json())