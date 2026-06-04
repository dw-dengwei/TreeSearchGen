from utils.extract_json import extract_json_from_response
import json
from pydantic import BaseModel, Field

class Size(BaseModel):
  length: float = Field(description="The length of the object", ge=2.0, le=10)
  width: float = Field(description="The width of the object", ge=1.5, le=10)
  height: float = Field(description="The height of the object", ge=3.0)


class RoomSizeResponseModel(BaseModel):
  reasoning: str = Field(description="The reasoning for the decision")
  size: Size = Field(description="The size of the room")

def fmt(room_desc):
  ret = {}
  ret['user_prompt'] = \
f"""
You are an expert interior designer tasked with analyzing a room layout. Based on the provided room type, room description, and the description of the room size.
 Your job is to estimate the size of the room.
Input:
- Room Description: {room_desc}

You need to estimate a resonable size for the room, based on the description of the size, formatted as (length, width, height) in meters.
"""


  def check(response):
    try:
      js = json.loads(extract_json_from_response(response))
    except Exception:
      raise Exception(f"Your last output has errors. Please fix it in this term: No valid JSON:\n{response}\n\n. Ensure your json is surround by ```json and ```, no '//' comments")
    if 'output' not in js.keys():
      raise Exception(f"Your last output has errors. Please fix it in this term: The output has no `output` key:\n{response}\n\n")
      
  ret['check_fn'] = check
  return ret

def get_room_size(room_desc, get_response_task):
  response = get_response_task(task='ROOM SIZE', **fmt(room_desc), response_model=RoomSizeResponseModel)
  response['size'] = [response['size']['length'], response['size']['width'], response['size']['height']]
  response = {"output": response}
  return response
