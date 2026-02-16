# This is a sample duck brain

# You are given:
#  - north: the obstacle to the north of mother duck
#  - south: the obstacle to the south of mother duck
#  - west: the obstacle to the west of mother duck
#  - east: the obstacle to the east of mother duck
#  - thought: the thought mother duck is remembering


def brain(north, south, east, west, thought)
    if thought == 0:
        # Check to see which direction we should follow, to our right
        return None, 'south'
    elif thought == 'south':
        # We are going south with a wall to our right (west)
        if south == 'x' and west == 'x':
            # Blocked to our west and south
            if east == 'x': # If blocked on all three sides, go to our left
               return 'N', 'north'
            else:
                return 'E', 'east'
        elif west == 'x':
            return 'S', 'south'
        else: # If west is not blocked, go west
            return 'W', 'west'
    elif thought == 'west':
        # We are going west with a wall to our right (north)
        if north == 'x' and west == 'x': # Blocked going west and north
           if south =='x': # Blocked south, go back
               return 'E', 'east'
           else:
               return 'S', 'south'
        elif north == 'x':
            return 'W', 'west'
        else: # north is not blocked, go north
            return 'N', 'north'
    elif thought == 'north':
        # We are going north with a wall to our right (east)
        if east == 'x' and north == 'x': # Blocked going east and north
           if west == 'x': # Blocked west go back
               return 'S', 'south'
           else:
               return 'W', 'west'
        elif east == 'x':
            return 'N', 'north'
        else:
            return 'E', 'east'
    elif thought == 'east':
        # Going east, with wall to our right (south)
        if south == 'x' and east == 'x':
           if north == 'x': # Blocked north, go back
               return 'W', 'west'
           else:
               return 'N', 'north'
        elif south == 'x':
            return 'E', 'east'
        else:
            return 'S', 'south'

