'''Simple Turtle interpreter'''
import turtle
import time
import colorsys

screensize = turtle.screensize

def goto(x, y):
    return [('goto', (x, y))]

def forward(x):
    return [('forward', x)]

def penup():
    return ['penup']

def pendown():
    return ['pendown']

def color(c):
    return [('color', c)]

def pensize(z):
    return [('pensize', z)]

def fill(c, l):
    return [('fill', (c, l))]

def circle(r, p=None):
    return [('circle', (r, p))]

def turn(d):
    DIRS = { 'right': ('right', 90), 'left': ('left', 90), 'around': ('left', 180) }
    if isinstance(d, str):
        if d in DIRS:
            return [DIRS[d]]
        else:
            raise ValueError(f"{d} is not a valid turn. Valid choices are: " + ", ".join(DIRS.keys()))
    elif isinstance(d, (int, float)):
        if d < 0:
            return [('left', -d)]
        else:
            return [('right', d)]

def wait(seconds):
    return [('wait', seconds)]

def pause(prompt):
    return [('pause', prompt)]

def clearscreen():
    return ['clearscreen']


def darken(color, percentage=10):
    '''Darken a color by a certain percent'''
    return color_transform(color, 1 - (percentage / 100.0))

def lighten(color, percentage=10):
    '''Lighten a color'''
    return color_transform(color, 1 + (percentage / 100.0))

def color_to_rgb(color):
    '''Transform a turtle color to rgb'''
    canvas = turtle.getcanvas()
    r, g, b = canvas.winfo_rgb(color)
    return (r / 65536.0, g / 65536.0, b / 65536.0)

def blend(color1, color2, weight=0.5):
    '''Returns a color intermediate between color1 and color2'''
    r1, g1, b1 = color_to_rgb(color1)
    r2, g2, b2 = color_to_rgb(color2)

    return rgb_to_hex(r1 * weight + r2 * (1 - weight),
                      g1 * weight + g2 * (1 - weight),
                      b1 * weight + b2 * (1 - weight))

def color_transform(color, factor):
    '''Scale a color's lightness'''

    # Convert color to RGB format
    r, g, b = color_to_rgb(color)

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Decrease the brightness/value by the provided percentage
    v = max(0, min(1.0, v * factor))

    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return rgb_to_hex(r, g, b)

def rgb_to_hex(r, g, b):
    '''Convert rgb (red, green, blue) values into hex'''
    darkened_rgb = (int(r * 255), int(g * 255), int(b * 255))

    # Convert to a format that turtle can ingest (hex string)
    darkened_color = '#{:02x}{:02x}{:02x}'.format(*darkened_rgb)

    return darkened_color

def draw(l, speed=0.5, immediate=False, animate=True, save_pauses=False, pause_template=None, clear=True, relative=False):
    if clear:
        turtle.clearscreen()
        turtle.home() # Displays cursor before anything, if pause is used initially

        turtle.tracer(not immediate)

    sx, sy = turtle.position()

    def dodraw(l):
        for c in l:
            if c == 'penup':
                turtle.penup()
            elif c == 'pendown':
                turtle.pendown()
            elif c == 'clear':
                turtle.clearscreen()
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
                    if clear:
                        turtle.tracer(True)
                    input()
                    if clear:
                        turtle.tracer(not immediate)
            elif c[0] == 'left':
                turtle.left(c[1])
            elif c[0] == 'right':
                turtle.right(c[1])
            elif c[0] == 'goto':
                x, y = c[1]
                if relative:
                    turtle.goto(x + sx, y + sy)
                else:
                    turtle.goto(x, y)
            elif c[0] == 'color':
                turtle.color(c[1])
            elif c[0] == 'pensize':
                turtle.pensize(c[1])
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
