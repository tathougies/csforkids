import functools
from difflib import get_close_matches
import inspect
import time

def say(words):
    print(words)
    time.sleep(0.5)

EXTRA_ACTIONS = {}
def start(fn, state, extra_actions=None):
    global EXTRA_ACTIONS
    if extra_actions is not None:
        EXTRA_ACTIONS = extra_actions
    else:
        EXTRA_ACTIONS = {}

    fn = functools.partial(fn, state)
    while fn:
        fn = fn()  # Trampoline
    print("Done")

def interact(items=None, directions=None, state=None, custom_actions=None):
    """Interactive fiction game interface where player can take items,
    move in directions, or use custom actions.
    """
    if items is None:
        items = {}
    if directions is None:
        directions = {}
    if custom_actions is None:
        custom_actions = {}

    def help_action():
        print("Available actions: take <item>, go <direction>, describe, help")
        extra = [*EXTRA_ACTIONS.keys(), *custom_actions.keys()]
        if extra:
            print("Other actions:", ", ".join(extra))

    while True:
        command = input("What would you like to do? ").strip().lower()
        if not command:
            print("I don't understand what you want to do.")
            continue

        words = command.split()
        action = words[0]
        target = " ".join(words[1:])

        if action == "take":
            match = get_close_matches(target, items.keys(), n=1, cutoff=0.6)
            if match:
                return functools.partial(items[match[0]], state)
            else:
                print(f"Cannot find item '{target}'.")
        elif action == "go":
            match = get_close_matches(target, directions.keys(), n=1, cutoff=0.6)
            if match:
                return functools.partial(directions[match[0]], state)
            else:
                print(f"Cannot go '{target}'.")
        elif action == "describe":
            frame = inspect.currentframe().f_back
            return functools.partial(frame.f_globals[frame.f_code.co_name], state)
        elif action == "help":
            help_action()
        elif command in custom_actions:
            return functools.partial(custom_actions[command], state)
        elif command in EXTRA_ACTIONS:
            EXTRA_ACTIONS[command](state)
        else:
            print("I don't understand what you want to do.")
