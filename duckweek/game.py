from dataclasses import dataclass, field
from typing import Optional, Any, Protocol
from enum import Enum

import asyncio
import dis
import functools
import sys
import time
import random
import ast
import math
import pathlib
import json

DUCK_SPRITE_FILENAME = 'duckweek/assets/duck.png'
SCALE_FACTOR = 2

# Default window starting size
START_W = 800
START_H = 800

# Height of the bottom bar, in pixels
INFO_BAR_HEIGHT = 200

MOVE_VELOCITY = 8
INITIAL_DUCK_VELOCITY = 10

# How many animation ticks per second
TIME_SPEED = 2

# DO NOT MODIFY ANYTHING PAST HERE
COND_BRANCH_OPS = set(['POP_JUMP_FORWARD_IF_FALSE', 'POP_JUMP_FORWARD_IF_TRUE',
                       'POP_JUMP_IF_TRUE', 'POP_JUMP_IF_FALSE',
                       'POP_JUMP_IF_NOT_NONE', 'POP_JUMP_IF_NONE',
                       'JUMP_IF_TRUE', 'JUMP_IF_FALSE'])



# GL stuff
GL_VERTEX_SHADER_SRC = '''
attribute vec2 a_position;
uniform vec4 pos;
uniform vec4 lakeOffset;
uniform vec2 sourceTileSize;
uniform int darken;

varying vec2 textureOffset;

void main() {
  vec2 sourcePos = pos.zw;
  vec2 tileSize = lakeOffset.zw;
  vec2 tilePos = pos.xy * tileSize;

  // tilePos is 0 -> 1 screen position
  // lakeOffset is 0 -> 1 offset of tile 0 in 0 -> screen width/height coords

  // Calculate the [-1,1] screen position of the tile
  tilePos *= 2.0;
  tilePos -= 1.0;

  // offset by lake offset
  tilePos.x -= lakeOffset.x * 2.0;
  tilePos.y -= lakeOffset.y * 2.0;

  // Now this is the top left corner of the tile
  // a_position is the tile relative position, (0,0) -> top left, (1,1) -> bottom right
  tilePos.x += a_position.x * tileSize.x * 2.0;
  tilePos.y += a_position.y * tileSize.y * 2.0;

  tilePos.y *= -1.0;
  textureOffset = vec2(sourcePos.x + a_position.x * sourceTileSize.x,
                       1.0 - (sourcePos.y + a_position.y * sourceTileSize.y));
//  textureOffset = a_position;
  gl_Position = vec4(tilePos, 0.0, 1.0);
}
'''
GL_FRAGMENT_SHADER_SRC = '''
varying vec2 textureOffset;

uniform int darken;
uniform sampler2D atlas;

void main() {
  vec4 c = texture2D(atlas, textureOffset);
  c.w -= float(darken) * 0.2;
  gl_FragColor = c;
}
'''

def hits_sdl_rect(p, r):
    x, y = p
    rx, ry, rw, rh = r
    dx = x - rx
    dy = y - ry

    return dx >= 0 and dx < rw and dy >= 0 and dy < rh

def ceildiv(a, b):
    return (a + b - 1) // b

class DuckState(Enum):
    ALIVE = 'ALIVE'
    STEPPING = 'STEPPING'
    COMPLETE = 'COMPLETE'
    DEAD = 'DEAD'
    STOPPED = 'STOPPED'

    @property
    def next_state(self):
        if self == DuckState.STEPPING:
            return DuckState.STOPPED
        else:
            return self

    @property
    def is_alive(self):
        '''Check if the current sate is considered alive'''
        return self == DuckState.ALIVE or self == DuckState.STEPPING

    @property
    def is_complete(self):
        '''Check if the duck is complete.'''
        return self == DuckState.DEAD or self == DuckState.COMPLETE

@dataclass
class DuckInput:
    # Obstacle in this direction
    north: str
    east: str
    south: str
    west: str

    duck_state: Any

    # Later levels let the duck drop feathers
    feather: Optional[str] = None

    def make_globals(self):
        return { 'north': self.north,
                 'east': self.east,
                 'south': self.south,
                 'west': self.west,
                 'thought': self.duck_state,
                 'feather': self.feather }

class Disasmed:
    __slots__ = ('instructions', 'insnmap', 'linemap')

    def __init__(self, code):
        self.instructions = list(dis.get_instructions(code))
        self.insnmap = {}
        self.linemap = {}

    def __getitem__(self, off):
        if off not in self.insnmap:
            for i in self.instructions:
                if i.offset == off:
                    self.insnmap[off] = i
                    break
        return self.insnmap[off]

    def insns_for_line(self, line):
        if line not in self.linemap:
            insns = [i for i in self.instructions if i.positions.lineno <= line and i.positions.end_lineno >= line]
            self.linemap[line] = insns
        return self.linemap[line]

DISASMED_CACHE = {}
@dataclass
class ThoughtPath:
    next_name_stored_at: Optional[int] = None
    next_direction_stored_at: Optional[int] = None
    branches_at: list[int] = field(default_factory=list)

class ThoughtTracker:
    '''Track how a thought was made'''
    def __init__(self, brain):
        self.brain = brain
        self.thought_path = ThoughtPath()

    def __call__(self, frame, event, arg):
        if frame.f_code is self.brain.byte_code and event == 'call':
            frame.f_trace_opcodes = True # Request opcode tracing
            return functools.partial(self.track_code, self._disasm(frame.f_code))

    def _disasm(self, c):
        global DISASMED_CACHE
        if c not in DISASMED_CACHE:
            DISASMED_CACHE[c] = Disasmed(c)
        return DISASMED_CACHE[c]

    def track_code(self, asm, frame, event, arg):
        if event == 'opcode':
            i = asm[frame.f_lasti]
            if i.opname == 'STORE_NAME':
                if i.argval == 'next_thought':
                    self.thought_path.next_name_stored_at = i.positions.lineno
                elif i.argval == 'next_direction':
                    self.thought_path.next_direction_stored_at = i.positions.lineno
            elif i.opname in COND_BRANCH_OPS:
                self.thought_path.branches_at.append(i.positions.lineno)


class Brain:
    def __init__(self, brain_file=None, source=None):
        assert brain_file is not None or source is not None, "Must be made with source or brain_file"
        if brain_file is not None:
            self.source_code = pathlib.Path(brain_file).read_text()
        else:
            brain_file = "<reloaded>"
            self.source_code = source
        self.brain_file = brain_file
        self._validate_ast(self.source_code)
        self.byte_code = compile(self.source_code, brain_file, "exec")

    def __call__(self, duck_input):
        gbls = duck_input.make_globals()
        tracker = ThoughtTracker(self)
        try:
            sys.settrace(tracker)
            exec(self.byte_code, gbls)
        finally:
            sys.settrace(None)
        if 'next_direction' not in gbls:
            print("You didn't set 'next_direction'")
        return gbls.get('next_direction'), gbls.get('next_thought', duck_input.duck_state), tracker.thought_path

    def _validate_ast(self, code):
        ast.parse(code, 'exec')

class Tileset:
    def __init__(self, filename):
        with open(filename, 'rt') as f:
            self.data = json.load(f)

        self.asset_dir = pathlib.Path(filename).parent

        self.tiles_per_row = self.data['imagewidth'] // self.data['tilewidth']

        if 'tiles' not in self.data:
            self.data['tiles'] = []

        self.animation_period = 1
        self.water_tiles = set()
        for t in self.data['tiles']:
            if 'animation' in t:
                self.animation_period = math.lcm(self.animation_period, len(t['animation']))

            # Parse properties
            for p in t.get('properties', []):
                if p['name'] == 'water' and p['value']:
                    self.water_tiles.add(t['id'] + 1)

        self._cache = {}

    @property
    def image_file(self):
        return self.asset_dir / pathlib.Path(self.data['image'])

    def is_water(self, tilenum):
        return tilenum in self.water_tiles

    @property
    def tile_width(self):
        return int(self.data['tilewidth'])

    @property
    def tile_height(self):
        return int(self.data['tileheight'])

    def __getitem__(self, n):
        if n in self._cache:
            return self._cache[n]
        else:
            row = (n - 1) // self.tiles_per_row
            col = (n - 1) % self.tiles_per_row
            # Check animation

            tile = (row, col)

            anim = None
            animframes = 1
            for tiledef in self.data['tiles']:
                if tiledef['id'] == n and 'animation' in tiledef:
                    anim = []
                    for t in tiledef['animation']:
                        animn = t['tileid'] - 1
                        anim.append((animn // self.tiles_per_row, animn % self.tiles_per_row))
                    animframes = len(anim)
                    tile = anim
                    break
            if anim is None:
                anim = tile

            self._cache[n] = (animframes, anim)
            return (animframes, anim)

class MapLayer:
    def __init__(self, map, data):
        self.map = map
        self.data = data
        self.draw_duck_after = False

        props = self.data.get('properties')
        if props is not None:
            for prop in props:
                if prop.get('name') == 'draw_duck_after' and prop.get('type') == 'bool' and isinstance(prop.get('value'), bool):
                    self.draw_duck_after = prop['value']

    def __getitem__(self, c):
        y, x = c
        if x < 0 or x >= self.map.cols or y < 0 or y >= self.map.rows:
            return 0
        return self.data['data'][y * self.map.cols + x]

class Map:
    def __init__(self, map_file):
        with open(map_file, 'rt') as f:
            self.data = json.load(f)

        self.layers = [MapLayer(self, d) for d in self.data['layers']]

        if all(not l.draw_duck_after for l in self.layers):
            self.layers[0].draw_duck_after = True # If no layer is specified as 'draw_duck_after', then draw after layer 0

    def make_tracker(self, x=None):
        return [x] * (self.rows * self.cols)

    def is_complete(self, tracker, tileset):
        baselayer = self.layers[0]
        return all(tracker[self.get_index((r, c))]
                   for r in range(self.rows)
                   for c in range(self.cols)
                   if tileset.is_water(baselayer[r,c]))

    def get_index(self, pos):
        r, c = pos
        return r * self.cols + c

    @property
    def baselayer(self):
        return self.layers[0]

    @property
    def rows(self):
        return self.data['height']

    @property
    def cols(self):
        return self.data['width']

    def get_water_tiles(self, tileset):
        baselayer = self.layers[0]
        return [(r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if tileset.is_water(baselayer[r, c])]

    def get_size(self):
        return self.rows, self.cols

    def get_screen_center(self, tile_size):
        return (self.cols * tile_size / 2, self.rows * tile_size / 2)

    def random_starting_pos(self, tileset):
        '''Choose a random starting point in a water tile, using tileset to determine which tile is water.'''
        water_tiles = self.get_water_tiles(tileset)
        assert len(water_tiles) > 0, "No pond found in this layer"
        return random.choice(water_tiles)

class GameDelegate(Protocol):
    '''Interface specification for things (usually renderers) that can respond to game events.'''

    def duck_game_state_changed(self, sender: 'Game', new_game_state: DuckState):
        '''Sent when the duck dies, comes to life, or is complete!'''
        pass

    def duck_input_changed(self, sender: 'Game', new_input: DuckInput, duck_dir: str, duck_pos: tuple[int,int]):
        '''Sent when the duck makes a new decision (i.e, its thoughts, senses, etc)'''
        pass

    def thought_path_changed(self, sender: 'Game'):
        '''Changes were made to the game's thought path'''
        pass

    def set_camera_bounds(self, topleft: tuple[int, int], bottomright: tuple[int, int]):
        '''Set camera bounds'''
        pass

@dataclass
class GameResult:
    game: 'Game'
    final_state: DuckState
    duck_pos: tuple[int, int]
    duck_tracker: list[Any]

    def find_unreachable(self) -> Optional[tuple[int, int]]:
        for r, c in self.game.current_map.get_water_tiles(self.game.tileset):
            if self.duck_tracker[self.game.current_map.get_index((r, c))] is None:
                return (r, c)
        return None

class GameRunner:
    '''runs a game and tells you what happens'''

    def __init__(self, game):
        self.game = game

    def __call__(self, max_steps=10000) -> GameResult:
        game = self.game.clone()
        game.duck_game_state = DuckState.ALIVE

        steps = 0
        while game.duck_game_state.is_alive:
            if steps > max_steps:
                break

            game.complete_step()

            steps += 1
        return GameResult(final_state=game.duck_game_state,
                          game=self.game,
                          duck_pos=game.duck_pos,
                          duck_tracker=game.duck_tracker)

class Game:
    __slots__ = ('brain', 'tileset', 'tile_size', '_current_map',
                 'duck_pos', 'next_duck_pos', 'duck_state', 'initial_duck_pos',
                 'duck_game_state', 'duck_dir', 'duck_tracker', 'thought_path',
                 'delegate')

    def __init__(self, brain=None, tileset=None, first_map=None):
        assert tileset is not None, "Must supply a tileset"
        assert tileset.tile_width == tileset.tile_height, "Tiles must be squares"
        self.tileset = tileset
        self.tile_size = tileset.tile_width
        self.brain = brain
        self.delegate = None
        self.thought_path = []
        self._current_map = None

        # Duck position
        self.duck_pos = (0, 0) # Current position
        self.next_duck_pos = (0, 0) # Position at the end of the current step

        self.duck_state = 0 # Duck's current memory
        self.duck_game_state = DuckState.STOPPED
        self.duck_dir = None
        self.duck_tracker = None

        self.current_map = first_map

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def reload_brain(self, brain: Brain):
        self.brain = brain
        self.reset_game()

    def set_duck_dir(self, nextdir):
        duckrow, duckcol = self.duck_pos
        self.duck_dir = nextdir
        if nextdir == 'N':
            self.next_duck_pos = (duckrow - 1, duckcol)
        elif nextdir == 'E':
            self.next_duck_pos = (duckrow, duckcol + 1)
        elif nextdir == 'S':
            self.next_duck_pos = (duckrow + 1, duckcol)
        elif nextdir == 'W':
            self.next_duck_pos = (duckrow, duckcol - 1)

    def complete_step(self):
        '''Moves the duck to the next position, and computes the next square to go to'''
        r, c = self.next_duck_pos
        # Mark the new position as being complete
        self.duck_tracker[self.current_map.get_index(self.next_duck_pos)] = True
        # Check complete
        next_duck_game_state = self.duck_game_state
        if self.current_map.is_complete(self.duck_tracker, self.tileset):
            next_duck_game_state = DuckState.COMPLETE
            self.duck_pos = self.next_duck_pos
        if not self.tileset.is_water(self.current_map.baselayer[r, c]):
            # Game complete. duck died
            next_duck_game_state = DuckState.DEAD
            self.next_duck_pos = self.duck_pos # DO not proceed
            self.duck_dir = None
        else:
            self.duck_pos = self.next_duck_pos

        if next_duck_game_state != self.duck_game_state:
            self.duck_game_state = next_duck_game_state
            if self.delegate is not None:
                self.delegate.duck_game_state_changed(self, self.duck_game_state)

        duck_input = self.get_duck_input()
        if self.duck_game_state.is_alive:
            nextdir, self.duck_state, thought_path = self.brain(duck_input)
            self.add_thought_path(duck_input, thought_path)
            self.set_duck_dir(nextdir)
            self.duck_game_state = self.duck_game_state.next_state

        if self.delegate is not None:
            self.delegate.duck_input_changed(self, duck_input, self.duck_dir, self.duck_pos)

    def add_thought_path(self, duck_input, thought_path):
        self.thought_path.append((duck_input, thought_path))
        if self.delegate is not None:
            self.delegate.thought_path_changed(self)

    def get_duck_input(self):
        '''Calculate the current DuckInput based on the game state'''
        duckrow, duckcol = self.duck_pos
        north_tile = self.current_map.baselayer[duckrow - 1, duckcol]
        east_tile  = self.current_map.baselayer[duckrow, duckcol + 1]
        south_tile = self.current_map.baselayer[duckrow + 1, duckcol]
        west_tile  = self.current_map.baselayer[duckrow, duckcol - 1]

        return DuckInput(north=' ' if self.tileset.is_water(north_tile) else 'x',
                         east=' '  if self.tileset.is_water(east_tile) else 'x',
                         south=' ' if self.tileset.is_water(south_tile) else 'x',
                         west=' '  if self.tileset.is_water(west_tile) else 'x',
                         duck_state=self.duck_state)

    @property
    def current_map(self):
        return self._current_map

    @current_map.setter
    def current_map(self, new_map):
        self._current_map = new_map
        self.next_duck_pos = self.duck_pos = self.current_map.random_starting_pos(self.tileset)
        self.initial_duck_pos = self.duck_pos
        self.reset_game()

    def reset_game(self, randomize_starting_position=False):
        '''Reset the game so the duck is in its old starting position.'''

        if randomize_starting_position:
            self.initial_duck_pos = self.current_map.random_starting_pos(self.tileset)

        self.duck_game_state = DuckState.STOPPED
        self.duck_state = 0
        self.duck_tracker = self.current_map.make_tracker()
        self.thought_path = []
        self.duck_pos = self.initial_duck_pos
        duck_input = self.get_duck_input()
        nextdir, self.duck_state, thought_path = self.brain(duck_input)
        self.add_thought_path(duck_input, thought_path)
        self.set_duck_dir(nextdir)

        if self.delegate is not None:
            self.delegate.duck_input_changed(self, duck_input, self.duck_dir, self.duck_pos)

class GameBackend(Protocol):
    def get_brain_renderer(self) -> Optional['BrainRenderer']:
        '''Get the brain renderer, if any'''
        pass

    def set_brain(self, brain_src):
        '''Set the brain source code.'''
        pass

    def reset_game(self):
        '''Reset the current game'''
        pass

    def screen_tile_size(self) -> int:
        '''Get the on screen tile size'''
        pass

    def screen_size(self) -> tuple[int, int]:
        '''Get screen size'''
        pass

    def update_info_bar(self, duck_input: DuckInput, duck_dir: str, duck_pos: tuple[int, int]):
        '''Update the info bar.'''
        pass

    def clear_lake(self):
        '''Clear the lake surface'''
        pass

    def start_lake_draw(self, lake_off: tuple[float, float]):
        '''Signal that we're going to start drawing the lake'''
        pass

    def draw_lake_sprite(self, tile_source: tuple[int, int], tile_pos: tuple[float, float], is_dark:bool = False):
        '''Draw a sprite at tile_source into the lake buffer at tile_pos'''
        pass

    def draw_duck(self, tile_source: tuple[int, int], tile_pos: tuple[float, float]):
        '''Draw the duck at the given position'''
        pass

    def complete_frame(self, lake_off: tuple[float, float]):
        '''Frame is complete'''
        pass

    def process_events(self, renderer: 'GameRenderer', dt: float):
        '''Process any pending events'''
        pass

    def should_continue(self) -> bool:
        '''Whether the game loop should continue.'''
        pass

class PygameTextLayout:
    def __init__(self, surface, x = 20, y = 0):
        self.surface = surface
        self.y = y
        self.x = x
        self.start_x = x
        self.lineheight = 10

        self.table = False
        self.table_start = y
        self.cur_column_x = 0
        self.cur_column_width = 0
        self.column_width = 0

    def start_table(self, padding=10):
        self.table = True
        self.cur_column_x = self.start_x
        self.y += self.lineheight + padding
        self.lineheight = 10
        self.x = self.start_x

        self.table_start = self.y
        self.cur_column_width = 0
        self.column_width = 0

    def end_table(self):
        self.table = False
        self.newline()

    def next_column(self, padding=10):
        self.y = self.table_start
        self.cur_column_x += self.column_width + padding
        self.x = self.cur_column_x
        self.cur_column_width = 0
        self.column_width = 0

    def newline(self, padding=3):
        self.x = self.start_x
        self.y += self.lineheight + padding
        self.lineheight = 10

        if self.table:
            self.cur_column_width = 0
            self.x = self.cur_column_x

    def indent(self, xoff):
        self.start_x += xoff

    def place(self, text_surface, xoff=0, nl=False):
        self.surface.blit(text_surface, (self.x + xoff, self.y))

        text_w, text_h = text_surface.get_size()
        # Calculate column width for table
        if self.table:
            self.cur_column_width += xoff + text_w
            self.column_width = max(self.cur_column_width, self.column_width)

        self.x += xoff + text_w
        self.lineheight = max(self.lineheight, text_h)

        if nl:
            self.newline()

class BrainRenderer(Protocol):
    '''A brain renderer displays the current brain and lets the user edit it.
    When running, code is highlighted to show what choices were made'''

    def report_syntax_error(self, err: SyntaxError):
        '''Report a syntax error'''
        pass

    def is_shown(self) -> bool:
        '''Get whether the renderer is shown'''
        pass

    def start(self):
        '''Start the brain renderer.'''
        pass

    def quit(self):
        '''Stop the brain renderer.'''
        pass

    def set_brain(self, brain: Brain):
        '''Set the brain source code'''
        pass

    def show(self):
        '''Show the source renderer'''
        pass

    def process_events(self, delegate):
        '''Process events and forward to delegate'''
        pass

class PygameBackend(GameBackend):
    def __init__(self, game: Game, source_renderer: BrainRenderer, show_info_bar = True):
        import pygame
        self.pygame = pygame
        self.show_info_bar = show_info_bar
        self.info_bar_height = INFO_BAR_HEIGHT if show_info_bar else 0
        pygame.init()
        pygame.display.set_caption('DuckBot!')

        self.brain_renderer = source_renderer

        self.game = game
        if self.brain_renderer is not None:
            self.brain_renderer.set_brain(self.game.brain)
        self.screen = pygame.display.set_mode(
            (START_W, START_H),
            pygame.RESIZABLE | pygame.DOUBLEBUF
        )
        self.lakesurface = None
        self.infosurface = None
        self.font = pygame.font.Font(None, 24)
        self.interned_text = {}
        self.running = True

        self.show_brain_rect = (0, 0, 0, 0)

        self.unscaled_atlas = pygame.image.load(self.game.tileset.image_file).convert_alpha()
        self.unscaled_duck_sprites = pygame.image.load(DUCK_SPRITE_FILENAME).convert_alpha() # TODO

        self.set_scale(SCALE_FACTOR)
        self.recalc_sizes()

        self.dirkeysdown = { pygame.K_UP: False, pygame.K_DOWN: False,
                             pygame.K_LEFT: False, pygame.K_RIGHT: False }

    def get_brain_renderer(self):
        return self.brain_renderer

    def screen_tile_size(self):
        return self.game.tile_size * self.scale

    def should_continue(self):
        return self.running

    def request_quit(self):
        self.running = False

    def set_brain(self, brain_src):
        # TODO reload brain
        try:
            brain = Brain(source=brain_src)
        except SyntaxError as e:
            # There was an error loading the Python
            if self.brain_renderer is not None:
                self.brain_renderer.report_syntax_error(e)
        else:
            self.game.reload_brain(brain)
            if self.brain_renderer is not None:
                self.set_brain(brain)

    def process_events(self, renderer, dt):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
            elif event.type == self.pygame.VIDEORESIZE:
                self.recalc_sizes()
            elif (event.type == self.pygame.KEYDOWN or event.type == self.pygame.KEYUP) and \
                 event.key in self.dirkeysdown:
                self.dirkeysdown[event.key] = event.type == self.pygame.KEYDOWN
            elif event.type == self.pygame.MOUSEBUTTONUP:
                x, y = self.pygame.mouse.get_pos()
                w, h = self.screen.get_size()

                y = y - h + self.info_bar_height
                if hits_sdl_rect((x, y), self.show_brain_rect):
                    if self.brain_renderer.is_shown():
                        self.brain_renderer.hide()
                    else:
                        self.brain_renderer.show()

        self.brain_renderer.process_events(self)

        # Calc velocity
        dcx, dcy = self.get_camera_velocity()
        tile_size = self.game.tile_size
        renderer.set_camera_position(renderer.camera_position[0] + dcy * tile_size * dt,
                                     renderer.camera_position[1] + dcx * tile_size * dt)

    def set_scale(self, scale_factor):
        self.scale = scale_factor
        self.atlas = self._scale(self.unscaled_atlas)
        self.duck_sprites = self._scale(self.unscaled_duck_sprites)

    def _scale(self, surface):
        w, h = surface.get_size()
        return self.pygame.transform.scale(surface, (self.scale * w, self.scale * h))

    def get_camera_velocity(self):
        '''Calculate the camera velocity based on which keys are down'''
        vx = 0
        vy = 0
        if self.dirkeysdown.get(self.pygame.K_UP):
            vy -= 1
        if self.dirkeysdown.get(self.pygame.K_DOWN):
            vy += 1
        if self.dirkeysdown.get(self.pygame.K_LEFT):
            vx -= 1
        if self.dirkeysdown.get(self.pygame.K_RIGHT):
            vx += 1
        return (vx * MOVE_VELOCITY, vy * MOVE_VELOCITY)

    def recalc_sizes(self):
        screen_w, screen_h = self.screen.get_size()
        logical_h = screen_h - self.info_bar_height
        new_infosize = (screen_w, self.info_bar_height)

        tile_size = self.screen_tile_size()
        new_lakesize = (screen_w + tile_size, screen_h + tile_size)

        if self.lakesurface is None or\
           self.lakesurface.get_size() != new_lakesize:
            self.lakesurface = self.pygame.Surface(new_lakesize, self.pygame.SRCALPHA | self.pygame.HWACCEL)

        if (self.infosurface is None or \
            self.infosurface.get_size() != new_infosize) and \
            self.show_info_bar:
            self.infosurface = self.pygame.Surface(new_infosize, self.pygame.HWACCEL)
            self.draw_info_surface()

        # Figure out camera bounds. The camera points to the middle tile
        min_needed_x = (screen_w // 2) // self.screen_tile_size()
        min_needed_y = (screen_h // 2) // self.screen_tile_size()
        self.set_camera_bounds((min_needed_x, min_needed_y),
                               (self.game.current_map.cols - min_needed_x, self.game.current_map.rows - min_needed_y))

    def set_camera_bounds(self, topleft, bottomright):
        if self.game.delegate is not None:
            self.game.delegate.set_camera_bounds(topleft, bottomright)

    def intern_text(self, text, color='black', background=None):
        '''Render text and save it so we don't have to re-render it'''
        key = (text, color, background)
        if key in self.interned_text:
            return self.interned_text[key]
        else:
            self.interned_text[key] = self.font.render(text, True, color, background)
            return self.interned_text[key]

    @staticmethod
    def center_text(dest, text, ypos, xoff, xwidth):
        '''Convenience function to center text in a surface'''
        x = xoff + xwidth / 2 - text.get_width() / 2
        dest.blit(text, (x, ypos))

    def update_info_bar(self, duck_input: DuckInput, duck_dir: str, duck_pos: tuple[int, int]):
        pass

    def clear_lake(self):
        self.lakesurface.fill((0,0,0))

    def start_lake_draw(self, lakeoff):
        pass # Not needed here

    def draw_lake_sprite(self, tile_source, tile_pos, is_dark=False):
        tile_size = self.game.tile_size * self.scale
        tile_source_x, tile_source_y = tile_source
        self.lakesurface.blit(self.atlas, (tile_source_x * tile_size, tile_source_y * tile_size),
                              (tile_pos[0] * tile_size, tile_pos[1] * tile_size, tile_size, tile_size))
        if is_dark:
            self.lakesurface.fill((180,180,180,255), (tile_source[0] * tile_size, tile_source[1] * tile_size, tile_size, tile_size),
                                  special_flags=self.pygame.BLEND_RGBA_MULT)

    def draw_duck(self, tile_source, tile_pos):
        tile_size = self.game.tile_size * self.scale
        self.lakesurface.blit(self.duck_sprites, (tile_source[0] * tile_size, tile_source[1] * tile_size),
                              (tile_pos[0] * tile_size, tile_pos[1] * tile_size, tile_size, tile_size))

    def screen_size(self):
        return self.screen.get_size()

    def complete_frame(self, lake_off):
        screen_w, screen_h = self.screen.get_size()
        info_bar_y = screen_h - self.info_bar_height

        self.screen.fill((0,0,0))
        self.screen.blit(self.lakesurface,
                         (0, 0), (lake_off[0], lake_off[1], screen_w, info_bar_y))
        if self.show_info_bar:
            self.screen.blit(self.infosurface, (0, info_bar_y))
        self.pygame.display.flip()

    def draw_info_surface(self):
        if not self.show_info_bar:
            return
        self.infosurface.fill((180, 180, 180))

        title = self.intern_text('DuckBot!')
        screen_w, info_h = self.infosurface.get_size()
        self.center_text(self.infosurface, title, 10, 0, screen_w)

        text = PygameTextLayout(self.infosurface, y=10 + title.get_size()[1] + 10)
        text.place(self.intern_text("Current Status: "))
        if self.game.duck_game_state == DuckState.ALIVE:
            text.place(self.intern_text("ALIVE", 'black', 'white'))
        elif self.game.duck_game_state == DuckState.DEAD:
            text.place(self.intern_text("DEAD", 'red', 'white'))
        else:
            text.place(self.intern_text("COMPLETE", 'darkgreen', 'white'))

        text.newline(10)
        duck_input_txt = self.intern_text("Mother Duck:")

        text.indent(60)
        text.start_table()
        text.place(self.intern_text('direction'), nl=True)
        text.place(self.intern_text('thought'), nl=True)
        text.place(self.intern_text('north'), nl=True)
        text.place(self.intern_text('east'), nl=True)
        text.place(self.intern_text('south'), nl=True)
        text.place(self.intern_text('west'), nl=True)
        text.next_column()

        eq = self.intern_text('=')
        text.place(eq, nl=True)
        text.place(eq, nl=True)
        text.place(eq, nl=True)
        text.place(eq, nl=True)
        text.place(eq, nl=True)
        text.place(eq, nl=True)
        text.next_column()

        text.place(self.intern_text(self.game.duck_dir or 'None'), nl=True)
        duck_input = self.game.get_duck_input()
        text.place(self.intern_text(repr(duck_input.duck_state)), nl=True)
        text.place(self.intern_text(duck_input.north), nl=True)
        text.place(self.intern_text(duck_input.east), nl=True)
        text.place(self.intern_text(duck_input.south), nl=True)
        text.place(self.intern_text(duck_input.west), nl=True)
        text.end_table()

        buttons_x = screen_w - 150
        self.show_brain_rect = (buttons_x, 10, 140, 60)
        self.infosurface.fill((225, 225, 225), self.show_brain_rect)
        if self.brain_renderer:
            self.center_text(self.infosurface, self.intern_text('Hide Brain' if self.brain_renderer.is_shown() else 'Show Brain'),
                             30, buttons_x, 140)

    def duck_input_changed(self, sender, duck_input, duck_dir, duck_pos):
        self.draw_info_surface()

class GlBackend(GameBackend):
    def __init__(self, game: Game, source_renderer: 'TkRenderer'):
        from OpenGL import GL
        self.gl = GL
        self.game = game
        self.tk_renderer = source_renderer

        if self.tk_renderer is not None:
            self.tk_renderer.set_brain(self.game.brain)

        self.scale = SCALE_FACTOR
        self.cur_texture = None
        self.initialized = False

    def duck_input_changed(self, *args):
        self.tk_renderer.update_state_from_game(self.game)
    def duck_game_state_changed(self, *args):
        self.tk_renderer.update_state_from_game(self.game)

    def get_brain_renderer(self):
        return self.tk_renderer

    def start_or_pause(self):
        if self.game.duck_game_state == DuckState.STOPPED:
            self.game.duck_game_state = DuckState.ALIVE
        elif self.game.duck_game_state == DuckState.ALIVE:
            self.game.duck_game_state = DuckState.STOPPED

    def set_brain(self, brain_src):
        try:
            brain = Brain(source=brain_src)
        except SyntaxError as e:
            self.tk_renderer.report_syntax_error(e)
        else:
            self.game.reload_brain(brain)

    def reset_game(self):
        self.game.reset_game(randomize_starting_position=True)

    def _init_opengl(self):
        import numpy as np
        import ctypes
        self.tk_renderer.make_current()

        gl = self.gl

        # Create the mesh
        self.mesh_array = gl.glGenBuffers(1)
        mesh = np.array([[0,0],
                         [1,0],
                         [0,1],
                         [1,1]], dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.mesh_array)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, mesh.nbytes, mesh, gl.GL_STATIC_DRAW)

        # Create the textures
        self.lake_texture, self.duck_texture = gl.glGenTextures(2)
        self._load_texture(self.lake_texture, self.game.tileset.image_file)
        self._load_texture(self.duck_texture, DUCK_SPRITE_FILENAME)

        # Create shader
        self.vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        self.fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

        self._compile_shader(self.vertex_shader, GL_VERTEX_SHADER_SRC)
        self._compile_shader(self.fragment_shader, GL_FRAGMENT_SHADER_SRC)

        self.shader_program = gl.glCreateProgram()
        gl.glAttachShader(self.shader_program, self.vertex_shader)
        gl.glAttachShader(self.shader_program, self.fragment_shader)
        gl.glLinkProgram(self.shader_program)
        status = gl.glGetProgramiv(self.shader_program, gl.GL_LINK_STATUS)
        if not status:
            log = gl.GetProgramInfoLog(self.shader_program)
            raise RuntimeError(f'Program link error:\n{log.decode()}')

        self.pos_uniform = gl.glGetUniformLocation(self.shader_program, b'pos')
        self.lake_off_uniform = gl.glGetUniformLocation(self.shader_program, b'lakeOffset')
        self.source_tile_size_uniform = gl.glGetUniformLocation(self.shader_program, b'sourceTileSize')
        self.darken_uniform = gl.glGetUniformLocation(self.shader_program, b'darken')

        gl.glUseProgram(self.shader_program)
        a_position = gl.glGetAttribLocation(self.shader_program, 'a_position')
        gl.glVertexAttribPointer(a_position, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(a_position)

        # Bind sampler to texture 0
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, 'atlas'), 0)

    def _compile_shader(self, shader, src):
        gl = self.gl
        gl.glShaderSource(shader, src)
        gl.glCompileShader(shader)
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not status:
            log = gl.glGetShaderInfoLog(shader)
            raise RuntimeError(f'Shader compile error:\n{log.decode()}')

    def _load_texture(self, gltex, file):
        from PIL import Image
        import numpy as np
        img = Image.open(file).convert("RGBA")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        iw, ih = img.size
        tex = np.asarray(img, dtype=np.uint8)
        tex = np.ascontiguousarray(tex)
        img.close()

        gl = self.gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, gltex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, iw, ih, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)

    def screen_tile_size(self):
        return self.scale * self.game.tile_size

    def screen_size(self):
        return self.tk_renderer.glcanvas_size()

    def update_info_bar(self, duck_input, duck_dir, duck_pos):
        pass

    def clear_lake(self):
        gl = self.gl
        gl.glClearColor(0,0,0,1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    def start_lake_draw(self, lake_off):
        if not self.initialized:
            self._init_opengl()
            self.initialized = True

        self.gl.glEnable(self.gl.GL_BLEND)
        self.gl.glBlendFunc(self.gl.GL_SRC_ALPHA, self.gl.GL_ONE_MINUS_SRC_ALPHA)

        self.gl.glUseProgram(self.shader_program)

        # Set tile size in [0,1] coords
        self._switch_to_lake_texture()

        screenw, screenh = self.screen_size()
        tile_size = self.screen_tile_size()
        self.gl.glUniform4f(self.lake_off_uniform,
                            lake_off[0] / screenw,
                            lake_off[1] / screenh,
                            tile_size / screenw,
                            tile_size / screenh)

    def _switch_to_lake_texture(self):
        self._switch_texture(self.lake_texture, 1/self.game.tileset.tile_width, 1/self.game.tileset.tile_height)

    def _switch_texture(self, tex, tilew, tileh):
        if self.cur_texture == tex:
            return
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, tex)
        self.gl.glActiveTexture(self.gl.GL_TEXTURE0)
        self.gl.glUniform1i(self.gl.glGetUniformLocation(self.shader_program, 'atlas'), 0)
        self.gl.glUniform2f(self.source_tile_size_uniform, tilew, tileh)
        self.cur_texture = tex

    def draw_lake_sprite(self, tile_pos, tile_source, is_dark=False):
        self.gl.glUniform4f(self.pos_uniform,
                            float(tile_pos[0]),
                            float(tile_pos[1]),
                            float(tile_source[0] / self.game.tileset.tile_width),
                            float(tile_source[1] / self.game.tileset.tile_height))
        self.gl.glUniform1i(self.darken_uniform, 1 if is_dark else 0)
        self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0, 4)

    def draw_duck(self, tile_pos, tile_source):
        self._switch_texture(self.duck_texture, 1/4, 1/4)
        self.gl.glUniform4f(self.pos_uniform,
                            float(tile_pos[0]),
                            float(tile_pos[1]),
                            float(tile_source[0] / 4),
                            float(tile_source[1] / 4))
        self.gl.glUniform1i(self.darken_uniform, 0)
        self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0, 4)
        self._switch_to_lake_texture()

    def complete_frame(self, lake_off):
        pass

    def process_events(self, renderer, dt):
        # Update camera position
        dcx, dcy = self.tk_renderer.get_camera_velocity()
        tile_size = self.game.tile_size
        renderer.set_camera_position(renderer.camera_position[0] + dcy * tile_size * dt,
                                     renderer.camera_position[1] + dcx * tile_size * dt)

        self.tk_renderer.process_events(self)

    def should_continue(self):
        return True # Not needed

class GameRenderer:
    '''Generic renderer for a game.'''

    def __init__(self, game: Game, backend: GameBackend):
        self.game = game
        self.delegate = self.game.delegate
        self.game.delegate = self

        self.backend = backend

        self.tile_animation_t = 0
        self.next_frame = 0
        self.duck_velocity = INITIAL_DUCK_VELOCITY

        self.camera_position = self.game.current_map.get_screen_center(self.game.tile_size * SCALE_FACTOR)
        self.camera_bounds = ((0, 0), (self.game.current_map.cols, self.game.current_map.rows))

    def set_camera_position(self, camx, camy):
        ((leftx, topy), (rightx, bottomy)) = self.camera_bounds
        tile_size = self.backend.screen_tile_size()
        camx = min(max(camx, leftx * tile_size), rightx * tile_size)
        camy = min(max(camy, topy * tile_size), bottomy * tile_size)
        self.camera_position = (camx, camy)

    def set_camera_bounds(self, topleft, bottomright):
        x, y = self.camera_position
        tile_size = self.backend.screen_tile_size()
        if bottomright[0] < topleft[0]:
            x = topleft[0] * tile_size
        else:
            x = min(max(x, topleft[0] * tile_size), bottomright[0] * tile_size)
        if bottomright[1] < topleft[1]:
            y = topleft[1] * tile_size
        else:
            y = min(max(y, topleft[1] * tile_size), bottomright[1] * tile_size)
        self.camera_bounds = (topleft, bottomright)
        self.camera_position = x, y

    def thought_path_changed(self, sender):
        if hasattr(self.backend, 'thought_path_changed'):
            self.backend.thought_path_changed(sender)
        if self.delegate is not None:
            self.delegate.thought_path_changed(sender)

    def duck_input_changed(self, sender, duck_input, duck_dir, duck_pos):
        if hasattr(self.backend, 'duck_input_changed'):
            self.backend.duck_input_changed(sender, duck_input, duck_dir, duck_pos)
        if self.delegate is not None:
            self.delegate.duck_input_changed(sender, duck_input, duck_dir, duck_pos)

    def duck_game_state_changed(self, sender, duck_game_state):
        if hasattr(self.backend, 'duck_game_state_changed'):
            self.backend.duck_game_state_changed(sender, duck_game_state)
        if self.delegate is not None:
            self.delegate.duck_game_state_changed(sender, duck_game_state)

    def redraw_lake(self, tile_anim_frame: int, lake_off: tuple[float, float]):
        '''Redraw the lake with the given tile anim frame'''
        camera_row, camera_col = self.camera_position
        screen_width, screen_height = self.backend.screen_size()

        min_x = camera_col - (screen_width // 2)
        max_x = min_x + screen_width

        min_y = camera_row - (screen_height // 2)
        max_y = min_y + screen_height

        tile_size = self.game.tile_size
        left_tile_x = int(min_x // (tile_size * SCALE_FACTOR))
        top_tile_y = int(min_y // (tile_size * SCALE_FACTOR))

        right_tile_x = ceildiv(max_x, (tile_size * SCALE_FACTOR))
        bottom_tile_y = ceildiv(max_y, (tile_size * SCALE_FACTOR))

        tile_width = int(right_tile_x - left_tile_x) + 1
        tile_height = int(bottom_tile_y - top_tile_y) + 1

        tileset = self.game.tileset

        duck_row, duck_col = self.game.duck_pos

        duck_col_off = None
        duck_row_off = None
        if duck_col >= left_tile_x and duck_col <= right_tile_x:
            duck_col_off = duck_col - left_tile_x
        if duck_row >= top_tile_y and duck_row <= bottom_tile_y:
            duck_row_off = duck_row - top_tile_y

        # The duck is also traveling a particular direction
        next_duck_row, next_duck_col = self.game.next_duck_pos

        full_frame = 1/self.duck_velocity
        frame_progress = (full_frame - self.next_frame)/full_frame

        duck_y_off = (next_duck_row - duck_row) * frame_progress
        duck_x_off = (next_duck_col - duck_col) * frame_progress

        self.backend.start_lake_draw(lake_off)
        for layer_ix, layer in enumerate(self.game.current_map.layers):
            for x in range(tile_width):
                for y in range(tile_height):
                    tile = layer[top_tile_y + y, left_tile_x + x]
                    if tile == 0:
                        continue
                    animframes, tiledef = tileset[tile]
                    if isinstance(tiledef, list):
                        tiledef = tiledef[tile_anim_frame % animframes]
                    sprite_y, sprite_x = tiledef
                    is_dark = False
                    if self.game.duck_tracker is not None and \
                       self.game.duck_tracker[self.game.current_map.get_index((top_tile_y + y, left_tile_x + x))]:
                        is_dark = True
                    self.backend.draw_lake_sprite((x,  y), (sprite_x, sprite_y), is_dark=is_dark)
            if layer.draw_duck_after and \
               duck_col_off is not None and duck_row_off is not None:
                 dir_to_row = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
                 self.backend.draw_duck((duck_col_off + duck_x_off, duck_row_off + duck_y_off),
                                        ((tile_anim_frame % 4), dir_to_row[self.game.duck_dir or 'S']))

    def game_loop_iter(self, dt: float):
        '''Run one iteration of the game loop, with dt seconds passing since the last iteration.'''
        self.tile_animation_t = (self.tile_animation_t + dt * TIME_SPEED) % self.game.tileset.animation_period
        tile_anim_frame = int(self.tile_animation_t)

        self.next_frame -= dt

        if self.next_frame <= 0:
            self.next_frame = 1 / self.duck_velocity
            self.game.complete_step()

        screen_w, screen_h = self.backend.screen_size()
        camera_left = int(self.camera_position[1] - screen_w // 2)
        tile_size = self.backend.screen_tile_size()
        camera_top = int(self.camera_position[0] - screen_h // 2)
        start_tile_x = camera_left // tile_size
        start_tile_y = camera_top // tile_size

        lake_off_x = camera_left - start_tile_x * tile_size
        lake_off_y = camera_top - start_tile_y * tile_size

        self.backend.clear_lake()
        self.redraw_lake(tile_anim_frame, (lake_off_x, lake_off_y))
        self.backend.complete_frame((lake_off_x, lake_off_y))

    def game_loop(self):
        asyncio.run(self.async_game_loop())

    async def async_game_loop(self, fps=30, sleep_fn=asyncio.sleep):
        if self.backend.get_brain_renderer() is not None:
            self.backend.get_brain_renderer().start()

        seconds_per_frame = 1 / fps
        last_time = time.monotonic()
        while self.backend.should_continue():
            frame_start = time.monotonic()
            self.backend.process_events(self, frame_start - last_time)
            self.game_loop_iter(frame_start - last_time)
            frame_end = time.monotonic()
            last_time = frame_start

            frame_calc = frame_end - frame_start
            frame_remaining = seconds_per_frame - frame_calc

            await sleep_fn(max(0,frame_remaining))

#def run_game(brain_file=None, tileset=None, maps=None):
#    '''Play the duckbot game'''
#    pygame.init()
#    pygame.display.set_caption('DuckBot!')
#
#    brain = Brain(brain_file)
#
#    # Create a resizable window at the given width and height
#    screen = pygame.display.set_mode(
#        (START_W, START_H),
#        pygame.RESIZABLE | pygame.DOUBLEBUF
#    )
#
#    # Set by recalc_sizes later
#    lakesurface = None
#    lake_dirty = True
#
#    def recalc_sizes():
#        '''Recalculate the logical game size from the current window size.'''
#        nonlocal screen, lakesurface, lake_dirty
#        screen_w, screen_h = screen.get_size()
#
#        logical_h = screen_h - INFO_BAR_HEIGHT
#
#        tilecount_x = ceildiv(screen_w, (SCALE_FACTOR * tile_size)) + 1
#        tilecount_y = ceildiv(logical_h, (SCALE_FACTOR * tile_size)) + 1
#
#        new_lakesize = (tilecount_x * tile_size, tilecount_y * tile_size)
#
#        if lakesurface is None or \
#           lakesurface.get_size() != new_lakesize:
#            lakesurface = pygame.Surface(new_lakesize, pygame.SRCALPHA | pygame.HWACCEL)
#            lake_dirty = True
#
#    recalc_sizes()
#
#    # Font stuff
#    font = pygame.font.Font(None, 24)
#    interned_text = {}
#    def intern_text(text):
#        '''Render text and save it so we don't have to re-render it'''
#        nonlocal interned_text
#        if text in interned_text:
#            return interned_text[text]
#        else:
#            interned_text[text] = font.render(text, True, 'black')
#            return interned_text[text]
#
#    def center_text(dest, text, ypos, xoff, xwidth):
#        x = xoff + xwidth / 2 - text.get_width() / 2
#        dest.blit(text, (x, ypos))
#
#    # UI concerns
#    running = True
#    clock = pygame.time.Clock()
#
#    current_map = None
#    camera_position = None # The position of the camera in screen space
#    duck_pos = (0, 0)
#    next_duck_pos = (0, 0)
#    duck_state = 0
#    duck_game_state = DuckState.ALIVE
#    duck_dir = None
#    duck_tracker = None
#
#    def reset_map(new_map):
#        '''Set a new map and adjust camera to the center'''
#        nonlocal current_map, camera_position, next_duck_pos, duck_pos, duck_tracker, duck_state
#        current_map = new_map
#        camera_position = new_map.get_screen_center(tile_size * SCALE_FACTOR)
#        next_duck_pos = duck_pos = current_map.random_starting_pos(tileset)
#        duck_game_state = DuckState.ALIVE
#        duck_state = 0
#        duck_tracker = current_map.make_tracker()
#        nextdir, duck_state = brain(get_duck_input())
#        set_duck_dir(nextdir)
#
#    def get_duck_input():
#        nonlocal duck_pos, current_map, duck_state
#        duckrow, duckcol = duck_pos
#        north_tile = current_map.baselayer[duckrow - 1, duckcol]
#        east_tile = current_map.baselayer[duckrow, duckcol + 1]
#        south_tile = current_map.baselayer[duckrow + 1, duckcol]
#        west_tile = current_map.baselayer[duckrow, duckcol - 1]
#
#        return DuckInput(north=' ' if tileset.is_water(north_tile) else 'x',
#                         east=' ' if tileset.is_water(east_tile) else 'x',
#                         south=' ' if tileset.is_water(south_tile) else 'x',
#                         west=' ' if tileset.is_water(west_tile) else 'x',
#                         duck_state=duck_state)
#
#    def redraw_lake(tile_frame):
#        '''Redraw the lake surface'''
#        nonlocal duck_pos, duck_dir, duck_sprites
#        camera_row, camera_col = camera_position
#        screen_width, screen_height = screen.get_size()
#
#        min_x = camera_col - (screen_width // 2)
#        max_x = min_x + screen_width
#
#        min_y = camera_row - (screen_height // 2)
#        max_y = min_y + screen_height
#
#        left_tile_x = int(min_x // (tile_size * SCALE_FACTOR))
#        top_tile_y = int(min_y // (tile_size * SCALE_FACTOR))
#
#        right_tile_x = ceildiv(max_x, (tile_size * SCALE_FACTOR))
#        bottom_tile_y = ceildiv(max_y, (tile_size * SCALE_FACTOR))
#
#        tile_width = int(right_tile_x - left_tile_x)
#        tile_height = int(bottom_tile_y - top_tile_y)
#
#        for layer_ix, layer in enumerate(current_map.layers):
#            for x in range(tile_width):
#                for y in range(tile_height):
#                    tile = layer[top_tile_y + y, left_tile_x + x]
#                    if tile == 0:
#                        continue
#                    animframes, tiledef = tileset[tile]
#                    if isinstance(tiledef, list):
#                        tiledef = tiledef[tile_frame % animframes]
#                    sprite_y, sprite_x = tiledef
#                    lakesurface.blit(tileset.atlas, (x * tile_size, y * tile_size),
#                                     (sprite_x * tile_size, sprite_y * tile_size, tile_size, tile_size))
#                    if layer_ix == 0 and duck_tracker is not None and duck_tracker[current_map.get_index((top_tile_y + y, left_tile_x + x))]:
#                        lakesurface.fill((180,180,180,255), (x * tile_size, y * tile_size, tile_size, tile_size), special_flags=pygame.BLEND_RGBA_MULT)
#
#        duck_row, duck_col = duck_pos
#
#        duck_col_off = None
#        duck_row_off = None
#        if duck_col >= left_tile_x and duck_col <= right_tile_x:
#            duck_col_off = duck_col - left_tile_x
#        if duck_row >= top_tile_y and duck_row <= bottom_tile_y:
#            duck_row_off = duck_row - top_tile_y
#
#        # The duck is also traveling a particular direction
#        next_duck_row, next_duck_col = next_duck_pos
#
#        full_frame = 1/duck_velocity
#        frame_progress = (full_frame - next_frame)/full_frame
#
#        duck_y_off = (next_duck_row - duck_row) * frame_progress * tile_size * SCALE_FACTOR
#        duck_x_off = (next_duck_col - duck_col) * frame_progress * tile_size * SCALE_FACTOR
#
#        if duck_col_off is not None and duck_row_off is not None:
#            # Todo draw moving
#            dir_to_row = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
#            lakesurface.blit(duck_sprites, (duck_col_off * tile_size + duck_x_off, duck_row_off * tile_size + duck_y_off),
#                             ( (tile_frame % 4) * tile_size, dir_to_row[duck_dir or 'S'] * tile_size, tile_size, tile_size))
#
#    reset_map(maps[0])
#
#    duck_velocity = INITIAL_DUCK_VELOCITY # How fast the duck should move
#
#    dirkeysdown = { pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False }
#    dt = 0.0
#    tile_animation_t = 0
#    next_frame = 0
#
#    duck_sprites =
#    while running:
#        for event in pygame.event.get():
#            if event.type == pygame.QUIT:
#                running = False
#            elif event.type == pygame.VIDEORESIZE:
#                recalc_sizes()
#            elif (event.type == pygame.KEYDOWN or event.type == pygame.KEYUP) and \
#                 event.key in dirkeysdown:
#                dirkeysdown[event.key] = event.type == pygame.KEYDOWN
#
#        if any(down for down in dirkeysdown.values()):
#            # Calc velocity
#            dcx, dcy = camera_velocity(dirkeysdown)
#            camera_position = (camera_position[0] + dcy * tile_size * dt,
#                               camera_position[1] + dcx * tile_size * dt)
#
#        old_tile_anim_frame = int(tile_animation_t)
#        next_tile_animation_t = (tile_animation_t + dt * TIME_SPEED) % tileset.animation_period
#        next_tile_anim_frame = int(next_tile_animation_t)
#        tile_animation_t = next_tile_animation_t
#
#        next_frame -= dt
#
#        if next_frame <= 0:
#            next_frame = 1 / duck_velocity
#            set_duck_pos(next_duck_pos)
#            if duck_game_state == DuckState.ALIVE:
#                nextdir, duck_state = brain(get_duck_input())
#                set_duck_dir(nextdir)
#
##        redraw_lake(next_tile_anim_frame)
#
#        screen_w, screen_h = screen.get_size()
#        info_bar_y = screen_h - INFO_BAR_HEIGHT
#
#        camera_left = int(camera_position[1] - screen_w // 2)
#        camera_top = int(camera_position[0] - screen_h // 2)
#        start_tile_x = camera_left // tile_size
#        start_tile_y = camera_top // tile_size
#
#        lake_off_x = camera_left - start_tile_x * tile_size
#        lake_off_y = camera_top - start_tile_y * tile_size
#
#        screen.fill((0,0,0))
#
#        screen.blit(pygame.transform.scale(lakesurface, (screen_w, info_bar_y)), (0, 0), (lake_off_x, lake_off_y, screen_w, info_bar_y))
#
#        info_bar_rect = (0, info_bar_y, screen_w, screen_h)
#        screen.fill((180,180,180), rect=(0, info_bar_y, screen_w, INFO_BAR_HEIGHT))
#        center_text(screen, intern_text('DuckBot!'), info_bar_y + 10, 0, screen_w)
#
#        pygame.display.flip()
#        dt = clock.tick(20) / 1000.0

def _tk_index(line, off):
    return f'{line + 1}.{off}'

class TkTextAnnotator:
    def __init__(self, text):
        import tkinter

        self.tk = tkinter
        self.text = text

        self.anns = {}

    def clear(self):
        for t in self.anns:
            self.text.tag_remove(t)
        self.anns = {}

    def annotate(self, start_line, start_off, end_line, end_off, msg):
        tag = f'ann_{start_line}_{start_off}_{end_line}_{end_off}'
        self.text.tag_add(tag, _tk_index(start_line, start_off), _tk_index(end_line, end_off))
        self.anns[tag] = msg

        self.text.tag_configure(tag, underline=True)

class JsInfoUpdater:
    def __init__(self, api, game):
        self.api = api
        self.game = game

    def set_camera_bounds(self, tl, br):
        pass

    def duck_game_state_changed(self, sender, new_game_state):
        self._update()

    def duck_input_changed(self, sender, new_input, duck_dir, duck_pos):
        self._update()

    def thought_path_changed(self, sender):
        self._update()

    def _update(self):
        self.api.updateGameInfo({'game_state': str(self.game.duck_game_state),
                                 'input': self.game.get_duck_input().make_globals(),
                                 'duck_pos': self.game.duck_pos})

class WasmBrainRenderer(BrainRenderer):
    def __init__(self, api):
        self.api = api

    def process_events(self, delegate):
        pass

    def report_syntax_error(self, err):
        pass

    def is_shown(self):
        return True

    def start(self):
        pass

    def quit(self):
        pass

    def set_brain(self, brain):
        self.api.setBrain(brain.source_code)

    def show(self):
        pass

class AsyncSuspended:
    '''Simple class to 'park' a coroutine for further processing.'''

    def __await__(self):
        value = yield None # What it seems like 'await self' returns
        return value

class TkRenderer(BrainRenderer):
    '''In-process renderer for brain source code with an optional opengl panel.'''

    def __init__(self, use_opengl=False):
        import tkinter
        from tkinter_gl import GLCanvas
        from tkinter.scrolledtext import ScrolledText
        from OpenGL import GL

        self.gl = GL
        self.tk = tkinter

        self._events = []

        self.root = root = tkinter.Tk()
        root.title("DuckBot!")

        # Configure grid
        if use_opengl:
            glarea = 0
            sourcearea = 1
            root.grid_columnconfigure(0, weight=1) # GL canvas area
            root.grid_columnconfigure((1,2), weight=0) # Source area
        else:
            sourcearea = 0
            root.grid_columnconfigure(0, weight=0)

        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure((1,2), weight=0)

        self.source_code = ScrolledText(root, wrap="none")
        self.source_code.grid(row=0, column=sourcearea, columnspan=2, sticky="nsew", padx=8, pady=4)

        self.controls = tkinter.Frame(self.root)
        self.controls.grid(row=1, column=sourcearea, sticky="nsew", padx=8, pady=4)

        self.controls.grid_rowconfigure((0,1), weight=1)
        self.controls.grid_columnconfigure((0,1), weight=1)

        self.reloadbrain = tkinter.Button(self.root, text="Reload Brain",
                                          command=self._reload_brain)
        self.reloadbrain.grid(row=1, column=sourcearea, sticky="ew", padx=4)
        self.startpause = tkinter.Button(self.root, text="Stop Duck",
                                         command=self._startpause_brain)
        self.startpause.grid(row=1, column=sourcearea+1, sticky="ew", padx=4)
        self.duck_state = None
        self.resetduck = tkinter.Button(self.root, text="Reset Game",
                                        command=self._reset_game)
        self.resetduck.grid(row=2, column=sourcearea, sticky="ew", padx=4)

        self.loadmap = tkinter.Button(self.root, text="Load Map",
                                      command=self._load_map)
        self.loadmap.grid(row=2, column=sourcearea+1, sticky="ew", padx=4)

        owner = self
        class MyGlCanvas(GLCanvas):
            def __init__(self):
                super().__init__(root)
                self.cont = None
                self.parent = self

            def draw(self):
                self.make_current()
                if self.cont is not None:
                    self.cont.send(None)
                    self.swap_buffers()

        if use_opengl:
            self.glarea = MyGlCanvas()
            self.glarea.grid(row=0, column=glarea, sticky="nsew", padx=8, pady=4)
            self.glarea.make_current()
            self.glarea.bind("<Configure>", self._glresized)

            self.stateframe = tkinter.Frame(self.root);
            self.stateframe.grid(row=1, column=glarea)

            self.root.bind_all("<KeyPress>", functools.partial(self._onglkey, True))
            self.root.bind_all("<KeyRelease>", functools.partial(self._onglkey, False))

            self.dirkeys = { 'Up': False,
                             'Down': False,
                             'Left': False,
                             'Right': False,
                             'Shift_L': False,
                             'Shift_R': False }

            self._glsize = (400, 400)
        else:
            self.glarea = None

        self.stateframe.grid_columnconfigure((0,1,2,3), weight=0)
        self.stateframe.grid_columnconfigure(4, weight=1)
        self.stateframe.grid_rowconfigure((0,1,2), weight=0)

        tkinter.Label(self.stateframe, text="thought =").grid(row=0, column=0, sticky="nsew", padx=8, pady=4)
        self.thoughtlabel = tkinter.Label(self.stateframe, text="")
        self.thoughtlabel.grid(row=0, column=1, columnspan=3)

        tkinter.Label(self.stateframe, text="north =").grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self.northlabel = tkinter.Label(self.stateframe, text="")
        self.northlabel.grid(row=1, column=1)

        tkinter.Label(self.stateframe, text="east =").grid(row=1, column=2, sticky="nsew", padx=8, pady=4)
        self.eastlabel = tkinter.Label(self.stateframe, text="")
        self.eastlabel.grid(row=1, column=3)

        tkinter.Label(self.stateframe, text="south =").grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        self.southlabel = tkinter.Label(self.stateframe, text="")
        self.southlabel.grid(row=2, column=1)

        tkinter.Label(self.stateframe, text="west =").grid(row=2, column=2, sticky="nsew", padx=8, pady=4)
        self.westlabel = tkinter.Label(self.stateframe, text="")
        self.westlabel.grid(row=2, column=3)

    def _reload_brain(self):
        brainsrc = self.source_code.get("1.0", self.tk.END)
        self._events.append(('brain', brainsrc))

    def _reset_game(self):
        self._events.append(('reset', ()))

    def _startpause_brain(self):
        self._events.append(('startpause', ()))

    def _load_map(self):
        from tkinter import filedialog, messagebox
        selected = filedialog.askopenfilename(title="Select map .json file",
                                              filetypes=[("Duckbot map files", "*.json")])
        if selected is not None:
            try:
                new_map = Map(selected)
            except:
                import traceback
                traceback.print_exc()
                messagebox.showwarning(title = "Ooops",message = "That was not a valid map file")
            else:
                self._events.append(('set_map', new_map))

    def process_events(self, delegate):
        for c, a in self._events:
            if c == 'brain':
                delegate.set_brain(a)
            elif c == 'quit':
                delegate.request_quit()
            elif c == 'reset':
                delegate.reset_game()
            elif c == 'startpause':
                delegate.start_or_pause()
            elif c == 'set_map':
                delegate.game.current_map = a

        self._events[:] = []

    def update_state_from_game(self, game):
        self._update_startpause(game)
        duck_input = game.get_duck_input()

        self.thoughtlabel.configure(text=repr(duck_input.duck_state))
        self.northlabel.configure(text=duck_input.north)
        self.eastlabel.configure(text=duck_input.east)
        self.southlabel.configure(text=duck_input.south)
        self.westlabel.configure(text=duck_input.west)

    def _gamecontrols_configure(self, state):
        self.reloadbrain.configure(state=state)
        self.resetduck.configure(state=state)
        self.loadmap.configure(state=state)

    def _update_startpause(self, game):
        if game.duck_game_state != self.duck_state:
            if not game.duck_game_state.is_complete and \
               game.duck_game_state != DuckState.ALIVE:
                self.startpause.configure(text="Start Duck")
                self._gamecontrols_configure(state="normal")
            elif game.duck_game_state == DuckState.ALIVE:
                self.startpause.configure(text="Stop Duck")
                self._gamecontrols_configure(state="disabled")
            elif game.duck_game_state == DuckState.COMPLETE:
                self.startpause.configure(text="Complete!")
                self._gamecontrols_configure(state="normal")
            else:
                self.startpause.configure(text="Duck Dead")
                self._gamecontrols_configure(state="normal")
            self.duck_state = game.duck_game_state

    def _glresized(self, i):
        self._glsize = (self.glarea.winfo_width(), self.glarea.winfo_height())
        self.gl.glViewport(0, 0, self._glsize[0], self._glsize[1])

    def _onglkey(self, state, e):
        if isinstance(e.widget, (self.tk.Entry, self.tk.Text)):
            return

        if e.keysym in self.dirkeys:
            self.dirkeys[e.keysym] = state

    def glcanvas_size(self):
        return self._glsize

    def make_current(self):
        if self.glarea is not None:
            self.glarea.make_current()

    def report_syntax_error(self, err: SyntaxError):
        print("TODO: report_syntax_error")

    def is_shown(self):
        return True

    def start(self):
        pass

    def quit(self):
        pass

    def set_brain(self, brain):
        self.source_code.delete("1.0", "end")
        self.source_code.insert("1.0", brain.source_code)

    def show(self):
        self.root.deiconify()

    def get_camera_velocity(self):
        '''Calculate camera velocity based on keys down'''
        vx = 0
        vy = 0
        if self.dirkeys.get('Up'):
            vy -= 1
        if self.dirkeys.get('Down'):
            vy += 1
        if self.dirkeys.get('Left'):
            vx -= 1
        if self.dirkeys.get('Right'):
            vx += 1

        if self.dirkeys.get('Shift_R') or self.dirkeys.get('Shift_L'):
            vx *= 5
            vy *= 5

        return (vx * MOVE_VELOCITY, vy * MOVE_VELOCITY)


    def _reschedule(self, renderer, game_loop_fn, secs):
        self.root.after(secs, game_loop_fn) # Schedule the loop to be run again in some amount of seconds

    def _game_loop(self, renderer):
        '''Run one game loop iteration of the current renderer'''
        renderer.game_loop_iter

    def run(self, renderer, *args, **kwargs):
        game_loop_continuation = None
        async_suspended = AsyncSuspended()
        def game_loop_iter():
            self.glarea.draw() # Requests the redraw

        def schedule_iter(secs):
            self.root.after(int(secs * 1000), game_loop_iter)
            return async_suspended

        kwargs['sleep_fn'] = schedule_iter
        game_loop_continuation = renderer.async_game_loop(*args, **kwargs)
        self.glarea.cont = game_loop_continuation
        self.root.mainloop()

class TkBrainRenderer(BrainRenderer):
    '''Out-of-process renderer just for Brain source code'''

    def __init__(self):
        import multiprocessing
        self.multiprocessing = multiprocessing
        self.started = False
        self.game_to_tk_queue = multiprocessing.Queue()
        self.tk_to_game_queue = multiprocessing.Queue()
        self.shown = False

    def show(self):
        if not self.shown:
            self.game_to_tk_queue.put(('show', ()))

    def set_brain(self, brain):
        self.game_to_tk_queue.put(('set_source', brain.source_code))

    def run(self, msg_queue, out_queue):
        import tkinter
        from tkinter.scrolledtext import ScrolledText

        root = tkinter.Tk()
        root.title("DuckBot!")

        root.withdraw()

        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=0)
        root.grid_rowconfigure(3, weight=1)
        root.grid_rowconfigure(4, weight=0)

        root.grid_columnconfigure(0, weight=1)

        tkinter.Label(root, text="Current Brain").grid(row=0, column=0)
        current = ScrolledText(root, wrap="none") #, font=("Menlo", 12))
        current.grid(row=1, column=0, sticky="nsew", padx=8, pady=(8,4))

        tkinter.Label(root, text="Editing Brain").grid(row=2, column=0)
        editing = ScrolledText(root, wrap="none")
        editing.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4,8))

        btns = tkinter.Frame()
        btns.grid(rows=4, column=0, sticky="ew", padx=8, pady=(4,8))
        btns.grid_columnconfigure((0,1,2,3), weight=1)

        editing_anns = TkTextAnnotator(editing)
        current.config(state='disabled')

        def request_close():
            out_queue.put(('quit', ()))
            root.destroy()

        def request_hide():
            out_queue.put(('hide', ()))
            root.withdraw()

        def send_shown_notification():
            out_queue.put(('shown', ()))

        def reload_brain():
            brainsrc = editing.get("1.0", tkinter.END)
            out_queue.put(('set_brain', (brainsrc,)))

        def reset_editing(src):
            editing_anns.clear()
            editing.delete("1.0", "end")
            editing.insert("1.0", src)

        def reset_current(src):
            current.config(state='normal')
            current.delete("1.0", "end")
            current.insert("1.0", src)
            current.config(state='disabled')

        def reset_brain():
            reset_editing(last_brain_source)

        def annotate_syntax_error(start_line, start_off, end_line, end_off, msg):
            editing_anns.annotate(start_line, start_off, end_line, end_off, msg)

        tkinter.Button(btns, text="Reset Brain",
                       command=reset_brain) \
               .grid(row=0, column=0, sticky="ew", padx=4)
        tkinter.Button(btns, text="Reload Brain",
                       command=reload_brain) \
               .grid(row=0, column=1, sticky="ew", padx=4)
        tkinter.Button(btns, text="Hide Brain Editor",
                       command=request_hide) \
               .grid(row=0, column=2, sticky="ew", padx=4)
        tkinter.Button(btns, text="Quit",
                       command=request_close) \
               .grid(row=0, column=3, sticky="ew", padx=4)

        last_brain_source = None
        def poll_queue():
            nonlocal last_brain_source
            import queue
            try:
                while True:
                    cmd, args = msg_queue.get_nowait()
                    if cmd == 'set_source':
                        reset_current(args)
                        reset_editing(args)

                        last_brain_source = args
                    elif cmd == 'show':
                        root.deiconify()
                        send_shown_notification()
                    elif cmd == 'hide':
                        root.withdraw()
                    elif cmd == 'syntax':
                        annotate_syntax_error(*args)
            except queue.Empty:
                pass
            root.after(30, poll_queue)

        def on_close():
            request_close()

        poll_queue()
        root.protocol('WM_DELETE_WINDOW', on_close)
        root.mainloop()

    def process_events(self, delegate):
        import queue
        try:
            while True:
                cmd, args = self.tk_to_game_queue.get_nowait()
                if cmd == 'quit':
                    delegate.request_quit()
                elif cmd == 'set_brain':
                    delegate.set_brain(args[0])
                elif cmd == 'hide':
                    self.shown = False
                elif cmd == 'shown':
                    self.shown = True
        except queue.Empty:
            pass

    def report_syntax_error(self, err):
        self.game_to_tk_queue.put(('syntax', (err.lineno, err.offset, err.end_lineno, err.end_offset, err.msg)))

    def is_shown(self):
        return self.shown

    def start(self):
        if self.started:
            return
        else:
            self.shown = False
            self.process = self.multiprocessing.Process(target=self.run, args=(self.game_to_tk_queue,self.tk_to_game_queue))
            self.process.start()
            self.started = True

    def quit(self):
        self.process.kill()
        self.process.join()
        del self.process

async def launch_html(brain_file, tileset_file, map_file, api):
    brain = Brain(brain_file)
    tileset = Tileset(tileset_file)
    game = Game(brain=brain, tileset=tileset, first_map=Map(map_file))
    game.delegate = JsInfoUpdater(api, game)
    wasm_renderer = WasmBrainRenderer(api)
    backend = PygameBackend(game, wasm_renderer, show_info_bar=False)
    renderer = GameRenderer(game, backend)
    await renderer.async_game_loop(10)

def launch_tkgl(brain_file, tileset_file, map_file):
    brain = Brain(brain_file)
    tileset = Tileset(tileset_file)
    game = Game(brain=brain, tileset=tileset, first_map=Map(map_file))
    print("FINAL STATE", GameRunner(game)().find_unreachable())
    tk_renderer = TkRenderer(use_opengl=True)
    backend = GlBackend(game, tk_renderer)
    renderer = GameRenderer(game, backend)
    tk_renderer.run(renderer)

def launch_pygame_tk(brain_file, tileset_file, map_file):
    brain = Brain(brain_file)
    tileset = Tileset(tileset_file)
    game = Game(brain=brain, tileset=tileset, first_map=Map(map_file))
    tk_renderer = TkBrainRenderer()
    backend = PygameBackend(game, tk_renderer)
    renderer = GameRenderer(game, backend)
    renderer.game_loop()

# This checks to see if this is being run as a game
if __name__ == '__main__':
    launch_tkgl('duckweek/sample_brain.py', 'duckweek/assets/tileset.json', 'duckweek/assets/maps/level1.json')
#    launch_pygame_tk('duckweek/sample_brain.py', 'duckweek/assets/tileset.json', 'duckweek/assets/maps/level1.json')
