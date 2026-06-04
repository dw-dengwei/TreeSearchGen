from utils.logger import logger
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

def create_response_model(choices: List[str], axis_type: str, use_image: bool):
    """
    Create a response model for either rows or columns.
    
    Args:
        choices: List of valid choices for the answer field
        axis_type: Either 'row' or 'column' to specify the type of response model
    
    Returns:
        A Pydantic model class for the response
    """
    logger.debug(f"offset choices: {choices}")
    if use_image:
      class ResponseModel(BaseModel):
          reasoning: str = Field(description="The reasoning for the decision")
          answer: Optional[List[str]] = Field(
              description=f"The {axis_type}s that the furniture should be placed",
              choices=choices
          )

          @field_validator('answer')
          @classmethod
          def validate_answer(cls, v: List[str], info) -> List[str]:
              if v is None:
                return v
              for item in v:
                if item not in choices:
                  if len(choices) == 0:
                    raise ValueError(f"Invalid choice: {item}. The valid emoji list is empty.")
                  raise ValueError(f"Invalid choice: {item}. Must be one of {choices}")
              return v
    else:
      class ResponseModel(BaseModel):
          reasoning: str = Field(description="The reasoning for the decision")
          answer: Optional[float] = Field(
              description=f"The center {axis_type} coordinate that the furniture should be placed",
          )
    
    return ResponseModel

def fmt(direction_desc, axis, anchor_name, close_to_the_wall, target_name, target_size, emoji_used, distance_desc, image_np):
  ret = {}
  ret['user_prompt'] = \
f"""
[Role]
You are a professional indoor designer. You know how to place furnitures in a room.

[Task]
Your task is determine the {axis}s that the furniture should be placed.

[Input]
Anchor Furniture Name: {anchor_name}, which is filled with red color and labled with its name.
Input Image: This image presents a floor of a room. The floor of the room is divided into a grid of cells. In this room, there is some furnitures. The furnitures are filled by red and labled with their names. In the free space, the cells are filled by emojis. Besides, the cells in the same {axis} share the same emoji. Around the floor, there are walls and boundaries. The walls are filled with the `wall` (which looks like a brick) emoji and the boundaries are filled with the `boundary` (which looks like a white circle) emoji. Both walls and boundaries cannot be occupied by furnitures. The walls and boundaries are different. The walls are the walls of a room and cannot be passed through. The boundaries are the boundaries of a region in the room and can be passed through.
Distance Requirement: {distance_desc}. Explain:
- if "adjacent_to": the new furniture should be placed close to (adjacent to) each anchor furniture.
- if "near": the new furniture should be placed near the anchor furniture but not close to (adjacent to) the anchor furniture.
- if "far": the new furniture should be  placed far from the anchor furniture.

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input, first assess the situation and explain how you are approaching the task.
2. **Break Down the Problem**: Identify key components of the task and explain the reasoning behind each step.
3. **Step-by-step Execution**: For each step, describe what you're doing, and why, providing reasoning behind decisions made. Ensure that intermediate outputs are included at each stage.
4. **JSON Outputs**: After you output the intermediate results. You need to output a JSON.

[Task Requirement]
1. I want to introduce a new furniture, "{target_name}", size = {target_size} {axis}s, into the room in the input image.
2. The new furniture should be placed {direction_desc} the anchor furniture, "{anchor_name}". You need to pay attention to the distance requirement.
3. The new furniture should be placed in the free space and should not overlap with any other furnitures.
4. Please describe the emojis that the furniture should be placed. Pay attention:
  - You can only describe the emojis with their names from the following list. The length of the list is {len(emoji_used)}:
  {emoji_used}
  - The length of emojis you provided must equals {target_size}, standing for the number of {axis}s the furniture need to be occupied.
  - The emojis you provided must be distinct.
  - The emojis you provided must be adjacent to each other.
{close_to_the_wall}
5. You should first output the reasoning for the decision.

6. If no suitable position is found, the answer should be None.
"""
      
  ret['check_fn'] = lambda _: None
  ret['image_np'] = image_np[:,:,::-1]
  return ret

def textual_fmt(direction_desc, axis, anchor_name, close_to_the_wall, target_name, target_size, emoji_used, distance_desc, text_layout):
  ret = {}
  ret['user_prompt'] = \
f"""
[Role]
You are a professional indoor designer. You know how to place furnitures in a room.

[Task]
Your task is determine the {axis} coordinates that the furniture should be placed.

[Input]
Anchor Furniture Name: {anchor_name}, which is filled with red color and labled with its name.
Textual Layout: {text_layout}
Distance Requirement: {distance_desc}. Explain:
- if "adjacent_to": the new furniture should be placed close to (adjacent to) each anchor furniture.
- if "near": the new furniture should be placed near the anchor furniture but not close to (adjacent to) the anchor furniture.
- if "far": the new furniture should be  placed far from the anchor furniture.

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input, first assess the situation and explain how you are approaching the task.
2. **Break Down the Problem**: Identify key components of the task and explain the reasoning behind each step.
3. **Step-by-step Execution**: For each step, describe what you're doing, and why, providing reasoning behind decisions made. Ensure that intermediate outputs are included at each stage.
4. **JSON Outputs**: After you output the intermediate results. You need to output a JSON.

[Task Requirement]
1. I want to introduce a new furniture, "{target_name}", size = {target_size} {axis}s, into the room in the input image.
2. The new furniture should be placed {direction_desc} the anchor furniture, "{anchor_name}". You need to pay attention to the distance requirement.
3. The new furniture should be placed in the free space and should not overlap with any other furnitures.
4. Please describe the emojis that the furniture should be placed. Pay attention:
  - You can only describe the emojis with their names from the following list. The length of the list is {len(emoji_used)}:
  {emoji_used}
  - The length of emojis you provided must equals {target_size}, standing for the number of {axis}s the furniture need to be occupied.
  - The emojis you provided must be distinct.
  - The emojis you provided must be adjacent to each other.
{close_to_the_wall}
5. You should first output the reasoning for the decision.

6. If no suitable position is found, the answer should be None.
"""
      
  ret['check_fn'] = lambda _: None
  ret['image_np'] = None
  return ret


def get_offset(direction_desc, axis, anchor_name, close_to_the_wall, target_name, target_size, emoji_used, distance_desc, image_np, text_layout, get_response_task, use_image=True):
    emoji_name = list(map(lambda s: f'  - {s}', emoji_used.keys()))
    emoji_name = '\n'.join(emoji_name)
    wall_policy_desc = ""
    if close_to_the_wall.lower() == 'yes':
        if 'top' in direction_desc:
            wall_policy_desc += "5. (IMPORTANT) It should be placed close to the **top most wall**."
        elif 'right' in direction_desc:
            wall_policy_desc += "5. (IMPORTANT) It should be placed close to the **right most wall**."
        elif 'bottom' in direction_desc:
            wall_policy_desc += "5. (IMPORTANT) It should be placed close to the **bottom most wall**."
        elif 'left' in direction_desc:
            wall_policy_desc += "5. (IMPORTANT) It should be placed close to the **left most wall**."

    choices = list(emoji_used.keys())
    ResponseModel = create_response_model(choices, axis.lower(), use_image)

    if len(choices) == 0:
      return {"output": {"answer": "None", "reason": "No suitable position is found."}}

    if use_image:
      ret = fmt(direction_desc, axis, anchor_name, close_to_the_wall, target_name, target_size, emoji_name, distance_desc, image_np)
    else:
      ret = textual_fmt(direction_desc, axis, anchor_name, close_to_the_wall, target_name, target_size, emoji_name, distance_desc, text_layout)
    response = get_response_task(task='OFFSET', **ret, response_model=ResponseModel)
    response = {"output": response}
    if response['output']['answer'] is None:
       response['output']['answer'] = "None"
    else:
      if use_image:
        response['output']['answer'] = list(map(lambda s: s.strip().lower(), response['output']['answer']))
    return response