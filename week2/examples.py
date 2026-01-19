import week2.draw as draw

def square_buggy(n):
  '''A buggy version of square()... uh oh!'''
  return \
    draw.forward(n) + draw.turn('right') + \
    draw.forward(n) + draw.turn('right') + \
    draw.forward(n) + draw.turn('right') + \
    draw.forward(n)

def square(n):
  return \
    draw.forward(n) + draw.turn('right') + \
    draw.forward(n) + draw.turn('right') + \
    draw.forward(n) + draw.turn('right') + \
    draw.forward(n) + draw.turn('right')
