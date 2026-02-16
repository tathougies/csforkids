# Welcome to DuckBot!

# DuckBot is a robot babysitter designed to care for ducklings.
# Mother Duck has left DuckBot in charge of her ducklings while she's away.
#
# But these naughty ducklings have caused DuckBot to go haywire and forget its programming.
#
# Your job is to write an algorithm to help DuckBot find the naughty ducklings, but be warned...
#
# The ducklings are very sneaky

# Your brain is written as a function named 'brain'
# You are given as arguments
#  - north: the obstacle to the north of DuckBot
#  - south: the obstacle to the south of DuckBot
#  - east: the obstacle to the east of DuckBot
#  - west: the obstacle to the west of DuckBot
#  - thought: the number or word DuckBot is remembering. Initially the number 0
#
#  The obstacle is either '*' if the square is free or 'x' if the square is an obstacle

# You return a direction and the next thought
#
# The direction can be one of 'N', 'E', 'S', 'W' for north, east, south, or west
# You can also use None, in which case DuckBot does nothing for that turn
#
# If you hit a wall or return incorrect output, then DuckBot will die!

def brain(north, south, east, west, thought):
    return 'E'
