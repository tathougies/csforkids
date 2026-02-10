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
    if east == 'x':
        next_thought = 'go_south'
    else:
        next_direction = 'E'
elif thought == 'next_column':
    if north == 'x':
        next_direction = 'W'
        next_thought = 'go_south'
    else:
        next_direction = 'N'
elif thought == 'go_south':
    if south == 'x':
        next_thought = 'next_column'
    else:
        next_direction = 'S'
