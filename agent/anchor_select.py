from pydantic import BaseModel, Field

def get_response_mode(objects):
  class AnchorResponseModel(BaseModel):
    reasoning: str = Field(description="The reasoning process of selecting the anchor object")
    anchor_name: str = Field(description="The name of the anchor object", choices=objects)
  return AnchorResponseModel


def fmt(room_desc, area, objects):
  ret = {}
  ret['user_prompt'] = \
f"""
You are tasked with classifying objects in a given room based on their role as either Anchor object or Other objects. An Anchor object is defined as a large item that represents the main function of a specified functional area. Other objects do not serve this primary role. Only 1 object can be classified as Anchor object.


Input:
- Room Description: {room_desc}
- Functional Area: {area['name']}
- Functional Area Description: {area['description']}
- Input Objects: 
{objects[0]}

Please follow these steps:

1. Read the provided room type and description.
2. Understand the functional area and its description.
3. Analyze the list of objects in the functional area.
4. You MUST select 1 Anchor object from the given objects list based on its size and relevance to the main function of the area. Even if none of the objects perfectly fits the role of an Anchor object, you must still select the one that best serves this purpose.
5. Create a JSON output that categorizes each object accordingly.

The JSON output should have the following structure:
"""

  ret['check_fn'] = None
  return ret

def get_anchor(room_desc, area, objects, get_response_task):
  response = get_response_task(task='GET ANCHOR', **fmt(room_desc, area, objects), response_model=get_response_mode(objects[2]))
  response = {"output": response}
  return response
