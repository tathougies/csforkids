from story import *

# Zeno's paradox

def zeno(state):
    say(f"You are {state['point'] * 100}% across the room")
    return interact(directions={'halfway': halfway}, state=state)

def halfway(state):
    state['point'] = (1.0 + state['point']) / 2.0
    return zeno(state)

INITIAL_STATE = {
    'point': 0.0
}

# DO NOT CHANGE
start(zeno, INITIAL_STATE)
