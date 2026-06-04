from random import choice

def get_abs_orientation(relation_type, ori_policy, anchor_orientation):
  if isinstance(relation_type, int):
    return [get_(relation_type, ori_policy, anchor_orientation)]
  else:
    return [get_(relation, ori_policy, anchor_orientation) for relation in relation_type]
    

def get_(relation_type, ori_policy, anchor_orientation):
  if ori_policy == 'A':
    if relation_type == 0:
      return 2
    elif relation_type == 1:
      return 3
    elif relation_type == 2:
      return 0
    elif relation_type == 3:
      return 1
  elif ori_policy == 'B':
    if relation_type == 0:
      return 0
    elif relation_type == 1:
      return 1
    elif relation_type == 2:
      return 2
    elif relation_type == 3:
      return 3
  elif ori_policy == 'C':
    if anchor_orientation == 0:
      return 2
    elif anchor_orientation == 1:
      return 3
    elif anchor_orientation == 2:
      return 0
    elif anchor_orientation == 3:
      return 1
  elif ori_policy == 'D':
    if anchor_orientation == 0:
      return 0
    elif anchor_orientation == 1:
      return 1
    elif anchor_orientation == 2:
      return 2
    elif anchor_orientation == 3:
      return 3
  elif ori_policy == 'E':
    return choice([0, 1, 2, 3])
