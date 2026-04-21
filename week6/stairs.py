from story import *

# A 'recursive' story

def stairs(state):
    if state['energy'] <= 0:
        say("You don't have enough energy to continue. You die")
        return
    if state['step'] >= state['max_steps']:
        say("You reached the top of the stairs")
        return

    say(f"You are at step {state['step']} with energy {state['energy']}")

    return interact(directions={'up': next_step}, state=state)

def next_step(state):
    state['step'] = state['step'] + 1
    state['energy'] = state['energy'] - 1

    return stairs(state)

INITIAL_STATE = {
    'energy': 5,
    'max_steps': 10,
    'step': 0
}

# DO NOT CHANGE
start(stairs, INITIAL_STATE)
