from story import *
import math
import random

# A simple story where you have to find a particular number

def search_rooms(state):
    rooms = state['rooms']
    current = state['current_index']
    current_room = rooms[current]
    goal = state['goal']

    if state['steps_left'] <= 0:
        say("You ran out of turns before finding the goal room.")
        return

    say(f"There are {len(rooms)} rooms in the tower.")
    say(f"You are now in room {current_room}.")
    say(f"The rooms are {state['rooms']}.")
    say(f"You have {state['steps_left']} turns left.")

    if current_room == goal:
        say("You found the goal room!")
        return

    if current_room < goal:
        say("The goal room has a larger number.")
    else:
        say("The goal room has a smaller number.")

    return interact(
        directions={
            'one up': one_up,
            'one down': one_down,
            'halfway up': halfway_up,
            'halfway down': halfway_down,
        },
        state=state
    )

def one_up(state):
    current = state['current_index']
    top = state['top_room']

    next = current + 1
    if next > top:
        next = top

    state['current_index'] = next
    return search_rooms(state)

def one_down(state):
    current = state['current_index']
    bottom = state['bottom_room']
    next = current - 1
    if next < bottom:
        next = bottom

    state['current_index'] = next
    return search_rooms(state)

def halfway_up(state):
    current = state['current_index']
    top = state['top_room']

    next_index = (current + top + 1) // 2
    if next_index == current and current < top:
        next_index = current + 1

    state['current_index'] = next_index
    state['steps_left'] = state['steps_left'] - 1

    return search_rooms(state)


def halfway_down(state):
    current = state['current_index']
    bottom = state['bottom_room']

    next_index = (current + bottom) // 2
    if next_index == current and current > bottom:
        next_index = current - 1

    state['current_index'] = next_index
    state['steps_left'] = state['steps_left'] - 1

    return search_rooms(state)

room_count = 32
rooms = sorted(random.sample(range(100, 1000000), room_count))
goal = random.choice(rooms)

INITIAL_STATE = {
    'rooms': rooms,
    'goal': goal,
    'current_index': len(rooms) // 2,
    'steps_left': math.ceil(math.log2(len(rooms))),
    'top_room': len(rooms) - 1,
    'bottom_room': 0
}

# DO NOT CHANGE
start(search_rooms, INITIAL_STATE)
