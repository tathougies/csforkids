import week2.draw as draw
import turtle
import time

import inspect
import inspect
import inspect

def call_with_supported_kwargs(fn, /, **kwargs):
    """
    Call `fn` using only keyword args *from the caller*, except:
      - if `fn` has positional-only params (and positional-or-keyword params),
        and kwargs contains a value for them, we move those into positional args
        in parameter order and remove them from kwargs.

    Extra kwargs are dropped unless fn accepts **kwargs.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    args = []
    consumed = set()

    # 1) Build positional args for parameters that MUST/SHOULD be positional
    for p in params:
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            # We are not generating *args from kwargs; stop building positional args here.
            break

        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if p.name in kwargs:
                args.append(kwargs[p.name])
                consumed.add(p.name)
            else:
                # If we already started supplying positional args, we can't skip a required
                # positional parameter in the middle.
                # If it's missing but has a default, we can stop adding positionals here.
                if args:
                    if p.default is inspect._empty:
                        raise TypeError(
                            f"{fn.__name__} missing required argument: {p.name!r}"
                        )
                    # default exists: don't push more positionals after this
                    break
                # If args is still empty, we can just not provide it positionally and let
                # normal calling rules / defaults handle it.

        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            # keyword-only stay in kwargs
            continue

        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs means we can keep extras
            continue

    # Remove consumed kwargs (moved into args)
    for name in consumed:
        kwargs.pop(name, None)

    # 2) If fn does NOT accept **kwargs, drop unknown kwargs
    if not has_varkw:
        allowed_kw = {
            p.name for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)
        }
        kwargs = {k: v for k, v in kwargs.items() if k in allowed_kw}

    # 3) Final call
    return fn(*args, **kwargs)

def lake_of(duck_fn, px_velocity=80):
  screen = turtle.Screen()

  last_frame = time.monotonic()
  t = 0
  while True:
    start_frame = time.monotonic()
    dt = start_frame - last_frame
    last_frame = start_frame

    turtle.clearscreen()
    turtle.tracer(False)
    turtle.home()

    w = turtle.window_width()
    h = turtle.window_height()
    turtle.penup()
    turtle.goto(-w/2, -h/6)

    turtle.pendown()
    turtle.color('deepskyblue')
    turtle.begin_fill()
    d = 0
    for x in range((w + 49) // 50):
      turtle.right(90)
      turtle.circle(25, 180)
      turtle.right(90)
      d += 50
    turtle.right(90)
    turtle.forward((5 * h)/6)
    turtle.right(90)
    turtle.forward(d)
    turtle.end_fill()

    turtle.penup()
    t += dt
    turtle.goto(w/2 - t * px_velocity, -h/6)

#    turtle.tracer(True)
    turtle.right(180)
    turtle.pendown()
    draw.draw(call_with_supported_kwargs(duck_fn, size=1, body_color='yellow', eye_color='green'), immediate=True, clear=False, animate=False, relative=True)

    turtle.penup()
    turtle.setheading(0)
    turtle.goto(w/2 - t * px_velocity + 100, -h/6)
    draw.draw(call_with_supported_kwargs(duck_fn, size=0.5, body_color='goldenrod', eye_color='black'), immediate=True, clear=False, animate=False, relative=True)

    turtle.penup()
    turtle.setheading(0)
    turtle.goto(w/2 - t * px_velocity + 200, -h/6)
    draw.draw(call_with_supported_kwargs(duck_fn, size=0.5, body_color='goldenrod', eye_color='blue'), immediate=True, clear=False, animate=False, relative=True)

    screen.update()
    time.sleep(1)
