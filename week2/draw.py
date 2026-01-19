'''Simple Turtle interpreter'''
import turtle
import time

def goto(x, y):
    return [('goto', (x, y))]

def forward(x):
    return [('forward', x)]

def penup(x):
    return ['penup']

def pendown(x):
    return ['pendown']

def color(c):
    return [('color', c)]

def fill(c, l):
    return [('fill', (c, l))]

def circle(r, p=None):
    return [('circle', (r, p))]

def turn(d):
    DIRS = { 'right': ('right', 90) }
    if isinstance(d, str):
        if d in DIRS:
            return [DIRS[d]]
        else:
            raise ValueError(f"{d} is not a valid turn. Valid choices are: " + ", ".join(DIRS.keys()))
    elif isinstance(d, (int, float)):
        if d < 0:
            return [('left', -d)]
        else:
            return [('right', 90)]


def wait(seconds):
    return [('wait', seconds)]

def pause(prompt):
    return [('pause', prompt)]

def draw(l, speed=0.5, animate=True, save_pauses=False, pause_template=None):
    turtle.clearscreen()
    turtle.home() # Displays cursor before anything, if pause is used initially

    def dodraw(l):
        for c in l:
            if c == 'penup':
                turtle.penup()
            elif c == 'pendown':
                turtle.pendown()
            elif c[0] == 'wait':
                if not animate:
                    continue
                time.sleep(c[1])
            elif c[0] == 'pause':
                if not animate:
                    continue
                else:
                    if save_pauses:
                        turtle.getcanvas().postscript(file='{}.eps'.format(pause_template.format(c[1])))
                    print(f"Waiting at {c[1]}. Hit Enter to continue")
                    input()
            elif c[0] == 'left':
                turtle.left(c[1])
            elif c[0] == 'right':
                turtle.right(c[1])
            elif c[0] == 'goto':
                x, y = c[1]
                turtle.goto(x, y)
            elif c[0] == 'color':
                turtle.color(c[1])
            elif c[0] == 'fill':
                color, ls = c[1]
                turtle.fillcolor(color)
                turtle.begin_fill()
                dodraw(ls)
                turtle.end_fill()
            elif c[0] == 'forward':
                turtle.forward(c[1])
            elif c[0] == 'circle':
                r, p = c[1]
                if p is not None:
                    p *= 360
                turtle.circle(r, p)
            else:
                raise ValueError(f"Unknown instruction {c}")

    dodraw(l)
