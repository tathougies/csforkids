# This is a sample duck brain

# You are given:
#  - north: the obstacle to the north of DuckBot
#  - south: the obstacle to the south of DuckBot
#  - west: the obstacle to the west of DuckBot
#  - east: the obstacle to the east of DuckBot
#  - thought: the number or word DuckBot is remembering

# You must set:
#  - next_direction: the direction DuckBot should travel
#  - next_thought: the number DuckBot should remember next

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
