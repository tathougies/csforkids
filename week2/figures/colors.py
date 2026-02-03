from turtle import Screen, Turtle, getcanvas

screen = Screen()
screen.setup(width=860, height=290)   # fixed logical size

t = Turtle()

# List of common turtle / Tk color names
COLORS = [
    # Basics
    "red", "orange", "yellow", "green", "blue", "purple",
    "black", "white", "gray",

    # Blues
    "cyan", "turquoise", "skyblue", "navy", "teal",
    "dodgerblue", "deepskyblue",

    # Greens
    "lime", "springgreen", "chartreuse", "darkgreen", "olive", "forestgreen", "seagreen",

    # Pinks / Purples
    "pink", "hotpink", "magenta", "violet", "orchid",
    "plum", "mediumpurple",

    # Reds / Oranges
    "salmon", "coral", "tomato", "orangered",
    "firebrick", "crimson",

    # Browns / Earth tones
    "brown", "sienna", "chocolate", "tan", "khaki",

    # Metallic / special
    "gold", "goldenrod", "amber", "darkorange", "peachpuff", "darkgoldenrod"
]

#COLORS = [
#    "red", "orange", "yellow", "green", "blue", "purple",
#    "black", "gray", "white",
#    "cyan", "turq
#    "cyan", "magenta", "pink", "brown", "black", "gray",
#    "gold", "silver", "navy", "skyblue", "turquoise",
#    "olive", "maroon", "violet", "coral",
#    "salmon", "khaki", "orchid", "plum"
#]

#turtle.colormode(255)
t.speed(0)
t.hideturtle()
t.penup()

# Grid settings
cols = 6
cell_x = 140
cell_y = 30
start_x = -420
start_y = 135

for i, color in enumerate(COLORS):
    row = i // cols
    col = i % cols

    x = start_x + col * cell_x
    y = start_y - row * cell_y

    # Draw color square
    t.goto(x, y)
    t.color('black')
    t.fillcolor(color)
    t.begin_fill()
    t.pendown()
    t.forward(20)
    t.right(90)
    t.forward(20)
    t.right(90)
    t.forward(20)
    t.right(90)
    t.forward(20)
    t.right(90)
    t.end_fill()
    t.penup()

    # Write label
    t.goto(x + 30, y - 20)
    t.write(color, align="left", font=("Arial", 13, "normal"))

getcanvas().postscript(file='turtle-colors.eps')
