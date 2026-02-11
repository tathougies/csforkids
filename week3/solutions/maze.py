# This is a sample duck brain

# You are given:
#  - north: the obstacle to the north of mother duck
#  - south: the obstacle to the south of mother duck
#  - west: the obstacle to the west of mother duck
#  - east: the obstacle to the east of mother duck
#  - memory: the number mother duck is remembering

# You must set:
#  - direction: the direction mother duck should travel
#  - memory: the number mother duck should remember next

if thought == 0:
    # Check to see which direction we should follow, to our right
    next_thought = 'south'
elif thought == 'south':
    # We are going south with a wall to our right (west)
    if south == 'x' and west == 'x':
        # Blocked to our west and south
        if east == 'x': # If blocked on all three sides, go to our left
           next_thought = 'north'
           next_direction = 'N'
        else:
           next_thought = 'east'
           next_direction = 'E'
    elif west == 'x':
        next_thought = 'south'
        next_direction = 'S'
    else: # If west is not blocked, go west
        next_thought = 'west'
        next_direction = 'W'
elif thought == 'west':
    # We are going west with a wall to our right (north)
    if north == 'x' and west == 'x': # Blocked going west and north
       if south =='x': # Blocked south, go back
          next_thought = 'east'
          next_direction = 'E'
       else:
          next_thought = 'south'
          next_direction = 'S'
    elif north == 'x':
        next_thought = 'west'
        next_direction = 'W'
    else: # north is not blokced, go north
        next_thought = 'north'
        next_direction = 'N'
elif thought == 'north':
    # We are going north with a wall to our right (east)
    if east == 'x' and north == 'x': # Blocked going east and north
       if west == 'x': # Blocked west go back
          next_thought = 'south'
          next_direction = 'S'
       else: 
           next_thought = 'west'
           next_direction = 'W'
    elif east == 'x':
        next_thought = 'north'
        next_direction = 'N'
    else:
        next_thought = 'east'
        next_direction = 'E'
elif thought == 'east':
    # Going east, with wall to our right (south)
    if south == 'x' and east == 'x':
       if north == 'x': # Blocked north, go back
          next_thought = 'west'
          next_direction = 'W'
       else:
          next_thought = 'north'
          next_direction = 'N'
    elif south == 'x':
       next_thought = 'east' 
       next_direction = 'E'
    else:
       next_thought = 'south'
       next_direction = 'S'
