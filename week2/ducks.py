from week2.draw import *

def foot1(sz):
    return color("orange") + forward(60 * sz) + \
        turn(30) + forward(10 * sz) + \
        turn(180) + forward(10*sz) + \
        turn(150) + forward(10*sz) + \
        turn(180) + forward(10*sz) + \
        turn(150) + forward(10*sz) + \
        turn(180) + forward(10*sz) + \
        turn(210)

def eye1(sz):
    return turn('around') + penup() + forward(10 * sz) + turn('left') + \
        pause('next') + \
        pendown() + color('black') + fill('white', circle(10 * sz)) + \
        circle(10 * sz, 1/2) + \
        fill('black', circle(3 * sz)) + turn('right')

def beak(sz):
    return fill('orange', forward(5 * sz) + turn(-110) + forward(5 * sz))

def triangle(n):
    return fill('orange', forward(5) + turn(120) + forward(5) + turn(120) + forward(5))

def duck1(size, body_color):
    return color(body_color) + fill(body_color, circle(30 * size) + circle(-50 * size)) + \
        penup() + turn('right') + \
        goto(20 * size, -80 * size) + pendown() + foot1(size) + penup() + \
        goto(-20 * size, -80 * size) + pendown() + foot1(size) + penup() + \
        goto(10 * size, 40 * size) + pendown() + pause('test') + eye1(size) + pause('testdone') + penup() + \
        goto(-10 * size, 40 * size) + pendown() + eye1(size) + penup() + \
        goto(0, 20 * size) + pendown() + beak(size)

def duck2_shape(size, return_to_x, return_to_y):
    return turn('around') + circle(size * 45, 1/2) + circle(size * 140, 13/64) + turn('around') + circle(-size * 15, 1/4) + goto(return_to_x, return_to_y)

def duck2(size, body_color):
    return color('black') + pensize(4) + fill(body_color, duck2_shape(size, 0, 0)) + \
        penup() + goto(-20 * size, 40 * size) + pendown() + \
        color('black') + \
        fill(body_color, circle(size * 30)) + penup() + \
        goto(20 * size, 0) + turn('around') + pendown() + \
        fill(blend(body_color, 'yellow'), duck2_shape(0.50 * size, 20, 0)) + penup() + turn(-30) + \
        goto(-40 * size, 0.5 * size) + \
        fill('orange', circle(-120 * size, 1/32) + turn(-45) + \
             circle(4 * size, 1/2) + turn(45) + forward(30 * size)) + \
        goto(-25 * size,10 * size) + fill('blue', circle(6 * size))
