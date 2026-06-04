def get_offset_direction(anchor_orientation, relation_type):
  if isinstance(relation_type, int):
    return [get_(anchor_orientation, relation_type)]
  else:
    return [get_(anchor_orientation, relation) for relation in relation_type]

def get_(anchor_orientation, relation_type):
  def get_angle(option):
    if option == 0: # forward
      return 0
    elif option == 1: # right
      return 90
    elif option == 2: # rearward
      return 180
    elif option == 3: # left
      return 270

  anchor_angle = get_angle(anchor_orientation)
  target_rel_angle = get_angle(relation_type)

  if (anchor_angle + target_rel_angle) % 360 == 0:
    return 0
  elif (anchor_angle + target_rel_angle) % 360 == 90:
    return 1
  elif (anchor_angle + target_rel_angle) % 360 == 180:
    return 2
  elif (anchor_angle + target_rel_angle) % 360 == 270:
    return 3