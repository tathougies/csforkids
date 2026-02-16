# This is a sample duck brain

# You are given:
#  - north: the obstacle to the north of DuckBot
#  - south: the obstacle to the south of DuckBot
#  - east: the obstacle to the east of DuckBot
#  - west: the obstacle to the west of DuckBot
#  - thought: the number or word DuckBot is remembering

def brain(north, south, east, west, thought):
  if thought == 'going_west':
    return None, 'south'
  if thought == 'south':
      if south == 'x':
          return None, 'next_column'
      if south == '*':
          return 'S', thought

  if thought == 'next_column':
    if north == 'x':
      return 'W', 'going_west'
    if north == '*':
      return 'N', thought

  if east == 'x':
      return None, 'south'

  return 'E', thought
