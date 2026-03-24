# Interactive Fiction
#
# Use this template to create your own interactive fiction stories.

from story import *

# PUT YOUR STORY HERE
def intro(state):
    say("You wake up in a tranquil forest covered in a gentle morning mist. The air is crisp, carrying the fragrant scent of pine needles.")
    say("Your mind is foggy, and as you stand up, you realize you don't remember how you got here or where 'here' even is.")
    return intro_scene(state)

def intro_scene(state):
    say("As you look around, you see a winding path ahead and a few intriguing objects resting on the forest floor, inviting you to choose.")
    items = {
        "lantern": pick_lantern,
        "map": pick_map
    }
    directions = {
        "north": intro_north,
        "south": intro_south
    }
    return interact(items=items, directions=directions, state=state)

def back_to_intro(state):
    say("You are back to where you started.")
    return intro_scene(state)

def intro_north(state):
    say("You decide to venture north, stepping carefully into the heart of the dense forest, where sunlight barely pierces the canopy above.")
    return rock_formation(state)

def intro_south(state):
    say("You choose to head south, where the trees begin to thin, revealing a sunlit clearing inviting you to explore its mysteries.")
    return clearing(state)

def pick_lantern(state):
    state['items'].append('lantern')
    say("You pick up the lantern, feeling its comforting weight in your hand, a beacon for whatever lies ahead.")
    directions = {
        "north": intro_north,
        "south": intro_south
    }
    return interact(directions=directions, state=state)

def pick_map(state):
    state['items'].append('map')
    say("You pick up the map, its crinkled edges hinting at untold adventures and secrets waiting to be uncovered.")
    directions = {
        "north": intro_north,
        "south": intro_south
    }
    return interact(directions=directions, state=state)

def rock_formation(state):
    say("As you journey further, you notice an unusual rock formation. It's strange how a particular stone seems out of place.")
    return interact(custom_actions={"look": look_at_rock}, directions={"south": back_to_intro}, state=state)

def look_at_rock(state):
    if 'key' in state['items']:
        say("You already took the key from here.")
    else:
        state['items'].append('key')
        say("Upon closer inspection, you find a small rusty key hidden under a rock. You decide to take it with you.")

    directions = {
        "south": back_to_intro
    }
    return interact(directions=directions, state=state)

def clearing(state):
    if 'key' in state['items']:
        say("In the clearing, there is an old wooden door set into a hillside. The key you found might fit.")
        return interact(custom_actions={"open door": use_key_on_door}, directions={"north": intro}, state=state)
    else:
        say("In the clearing, you see an old wooden door set into a hillside, but it seems to be locked.")
        directions = {
            "north": intro,
            "south": through_door
        }
        return interact(directions=directions, state=state)

def use_key_on_door(state):
    if 'key' in state['items']:
        say("You use the rusty key on the door, and with a satisfying click, it opens into darkness. Who knows what lies beyond?")
        state['door1'] = 'unlocked'
    else:
        say("You need a key to open this door.")
    directions = {
        "north": back_to_intro,
        "south": through_door
    }
    return interact(directions=directions, state=state)

def through_door(state):
    if state['door1'] == 'locked':
        say("The door is locked")
        return interact(directions={'north': back_to_intro}, state=state)
    else:
        return ice_cream(state)

def ice_cream(state):
    say("You find a magic land full of ice cream and lollipops. There's a unicorn pawing at you, as if it wants to take you somewhere.")

    # Now, it's your turn to do something fun!
    interact(directions={'north': back_to_intro}, state=state)

# Helper functions

def show_backpack(state):
    print("The following items are in your backpack:")
    print("\n".join(state['items']))

# STATE
INITIAL_STATE = {
    'items': [],
    'party': [],
    'health': 100,

    'door1': 'locked'
}

# Run

start(intro, INITIAL_STATE, extra_actions={
    'backpack': show_backpack
})
