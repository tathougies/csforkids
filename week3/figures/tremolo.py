from pyx import canvas, path, graph, text, style, color, deco, path, unit
import math

cycles = 3

# text.set(engine=text.LatexEngine, cmd=["xelatex", "-interaction=nonstopmode", "-halt-on-error"], copyinput="pyx_debug.tex", usefiles=["pyx_debug.log"])
# text.preamble(r"""
# \usepackage{fontspec}
# \setmainfont[
#   Path = ../../Book/Fonts/SourceSans3/,
#   Extension = .ttf
# ]{SourceSans3-Regular}
# """)

# --- styles ---
dashed = style.linestyle(style.linecap.round, [style.dash([2, 2])])
thin = style.linewidth.thin

# A small helper to draw a labeled callout box + connector
def callout(g, yvalue, label, x_from=0.0, x_to=2*math.pi, x_text=2*math.pi, y_text=2):
    """
    Draw a dashed horizontal line at y=yvalue from x_from..x_to,
    then connect to a labeled box placed at (x_box, yvalue).
    Coordinates are in graph (data) coordinates.
    """
    # dashed guide line
    g.stroke(path.path(path.moveto(*g.pos(x_from, yvalue)), path.lineto(*g.pos(x_to, yvalue))),
             [thin, style.linestyle.dashed, color.gray(0.4)])

    # connector from end of dashed line to the box
#             [thin, color.gray(0.4)])

    # label text (in graph coords), with a framed white-ish box feel
    b = text.text(g.pos(x_text, y_text)[0], g.pos(x_text, y_text)[1], label, [
        text.valign.middle,
        text.halign.left])
    bpath = b.bbox().enlarged(3 * unit.x_pt).path()
    g.stroke(path.path(path.moveto(*g.pos(x_to, yvalue)),
                       path.lineto(b.bbox().left(), b.bbox().top())),
             [thin, color.gray(0.4)])
    g.draw(bpath, [deco.filled([color.gray(0.9)]), deco.stroked([thin, color.gray(0.2)])])
    g.insert(b)
#        deco.filled([color.gray(1.0)]),   # fill behind text
#        deco.stroked([thin, color.gray(0.2)]),  # border
#        deco.bboxmargin(0.08),            # padding
#    ])

g = graph.graphxy(
    width=10,
    height=4,
    x=graph.axis.linear(min=0, max=1, title="time"),
    y=graph.axis.linear(min=-1.2, max=1.2, title="tone"))

g.plot(graph.data.function("y(x) = sin(2*pi*6*x)", context={'pi': math.pi}, points=1000))

callout(g, 1.0, "peak at 1", x_from=1/24.0, x_to=5/24.0, x_text=8/24.0, y_text=1)
callout(g, -1.0, "trough at -1", x_from=3/24.0, x_to=7/24.0, x_text=8/24.0, y_text=-1)

g.writePDFfile("week3/figures/tremolo-base.pdf")

g = graph.graphxy(
    width=10,
    height=2,
    x=graph.axis.linear(min=0, max=1, title="time"),
    y=graph.axis.linear(min=-0.5, max=0.5, title="tone"))

g.plot(graph.data.function("y(x) = sin(2*pi*6*x) * 0.1", context={'pi': math.pi}, points=1000))

callout(g, 0.1, "peak at 0.1", x_from=1/24.0, x_to=5/24.0, x_text=8/24.0, y_text=0.5)
callout(g, -0.1, "trough at -0.1", x_from=3/24.0, x_to=7/24.0, x_text=8/24.0, y_text=-0.5)

g.writePDFfile("week3/figures/tremolo-transformed.pdf")

g = graph.graphxy(
    width=10,
    height=2,
    x=graph.axis.linear(min=0, max=1, title="time"),
    y=graph.axis.linear(min=-1.2, max=1.2, title="tone"))

g.plot(graph.data.function("y(x) = sin(2*pi*6*x) * 0.1 + 1", context={'pi': math.pi}, points=1000))
g.plot(graph.data.function("y(x) = (sin(2*pi*6*x) * 0.1 + 1) * sin(2 * pi * 19 * x)", [graph.style.line([color.rgb.blue])], context={'pi': math.pi}, points=1000))

g.writePDFfile("week3/figures/tremolo-final.pdf")
