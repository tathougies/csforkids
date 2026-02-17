# This is a comment. Python ignores anything after the #
# Don't delete this next line

# Run this file to draw your duck
from . import draw
from .example import lake_of

# Define your duck here:
def myduck():
  return draw.circle(10)

draw.draw(myduck)

