import random
import glfw
import glm
import time
import math
import functools
import OpenGL.GLES2 as gl # gl* functions
from OpenGL.EGL import * # egl* functions
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "my_data_file.dat"

with open(Path(__file__).resolve().parent / "branch_shader.glsl", "rt") as f:
    BRANCH_FRAGMENT_SHADER = f.read()

_GL_INITIALIZED = False
def init_gl():
    global _GL_INITIALIZED
    if not _GL_INITIALIZED:
        do_init_gl()
    _GL_INITIALIZED = True

def do_init_gl():
    if not glfw.init():
        raise RuntimeError("glfw.init failed")

    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)

class EGLWindow(object):
    def __init__(self, render, title, listener=None):
        self.renderfunc = render
        self.window = None
        self.title = title
        self._listener = listener
        self._update_listener()

        init_gl()

        self._create()

    def _create(self):
        self.window = glfw.create_window(800, 600, self.title, None, None)
        if not self.window:
            raise RuntimeError("Could not create window: GLES not vaailable")

    def render(self):
        glfw.make_context_current(self.window)

        w, h = self.window_size()
        data = { 'width': w,
                 'height': h }
        self.renderfunc(**data)

    @property
    def listener(self):
        return self._listener

    print("LISTENER", listener.setter)
    @listener.setter
    def set_listener(self, l):
        self._listener = l
        self._update_listener()

    def _update_listener(self):
        if self._listener is not None:
            if hasattr(self._listener, 'key_callback'):
                glfw.set_key_callback(self.window, functools.partial(self._listener.key_callback, self))
            if hasattr(self._listener, 'mouse_pos_callback'):
                glfw.set_cursor_pos_callback(self.window, functools.partial(self._listener.mouse_pos_callback, self))
            if hasattr(self._listener, 'mouse_button_callback'):
                glfw.set_mouse_button_callback(self.window, functools.partial(self._listener.mouse_button_callback, self))

    def window_size(self):
        return glfw.get_framebuffer_size(self.window)

    def mainloop(self, fps=30): # Limit to 30 fps
        seconds_per_frame = 1.0 / fps

        while not glfw.window_should_close(self.window):
            start = time.perf_counter()
            w, h = self.window_size()

            glfw.make_context_current(self.window)
            gl.glViewport(0, 0, w, h)
            gl.glClearColor(0.53, 0.81, 0.92, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            self.render()

            glfw.swap_buffers(self.window)
            end = time.perf_counter()
            elapsed = end - start
            if elapsed > seconds_per_frame:
                glfw.poll_events()
            else:
                glfw.wait_events_timeout(seconds_per_frame - elapsed)

    def __enter__(self):
        '''Simple context manager that makes the gles2 context current inside.'''
        glfw.make_context_current(self.window)

    def __exit__(self, type, value, tb):
        return # Doesn't matter

class GLESShader(object):
    def __init__(self, prog, ty=gl.GL_VERTEX_SHADER):
        if isinstance(prog, str):
            prog = prog.encode('utf-8')
        self.shader = gl.glCreateShader(ty)
        gl.glShaderSource(self.shader, prog)
        gl.glCompileShader(self.shader)

        status = gl.glGetShaderiv(self.shader, gl.GL_COMPILE_STATUS)
        if not status:
            log = gl.glGetShaderInfoLog(self.shader)
            raise RuntimeError(f"Shader compile error:\n{log.decode()}")

    def __del__(self):
        if hasattr(self, 'shader'):
            gl.glDeleteShader(self.shader)

class GLESProgram(object):
    def __init__(self, vprog, fprog=None):
        if isinstance(vprog, str):
            vprog = GLESShader(vprog, gl.GL_VERTEX_SHADER)
        if isinstance(fprog, str):
            fprog = GLESShader(fprog, gl.GL_FRAGMENT_SHADER)

        self.vprog = vprog
        self.fprog = fprog
        self.attribs = GLESProgramAttrs(self)
        self.uniforms = GLESProgramUniforms(self)
        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, self.vprog.shader)
        if self.fprog is not None:
            gl.glAttachShader(self.program, self.fprog.shader)
        gl.glLinkProgram(self.program)

        status = gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)
        if not status:
            log = gl.glGetProgramInfoLog(self.program)
            raise RuntimeError(f"Program link error:\n{log.decode()}")

    def __del__(self):
        if hasattr(self, 'program'):
            gl.glDeleteProgram(self.program)

    def __enter__(self):
        gl.glUseProgram(self.program)

    def __exit__(self, t, v, tb):
        return

class GLESProgramVars(object):
    def __init__(self, program):
        self.program = program
        self._cache = {}

    def __getattr__(self, nm):
        if nm in self._cache:
            return self._cache[nm]
        else:
            attr = self._cache[nm] = self._get(nm)
            return attr

class GLESProgramAttrs(GLESProgramVars):
    def __init__(self, program):
        super().__init__(program)

    def _get(self, nm):
        loc = gl.glGetAttribLocation(self.program.program, nm)
        return GLESProgramAttr(self.program, loc)

class GLESProgramUniforms(GLESProgramVars):
    def __init__(self, program):
        super().__init__(program)

    def _get(self, nm):
        loc = gl.glGetUniformLocation(self.program.program, nm.encode('ascii'))
        return GLESProgramUniform(self.program, loc)

class GLESProgramVar(object):
    __slots__ = ('program', 'loc',)

    def __init__(self, program, loc):
        self.program = program
        self.loc = loc

class GLESProgramAttr(GLESProgramVar):
    pass

class GLESProgramUniform(GLESProgramVar):
    pass

class GLESBuffer(object):
    def __init__(self, data=None):
        self.vbo = gl.glGenBuffers(1)
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        with self:
            gl.glBufferData(self.gltype, data.nbytes, data, gl.GL_STATIC_DRAW)

    def __enter__(self):
        gl.glBindBuffer(self.gltype, self.vbo)
    def __exit__(self, t, v, tb):
        pass

class GLESVertexBuffer(GLESBuffer):
    gltype = gl.GL_ARRAY_BUFFER
    def __init__(self, data):
        super().__init__(data)

class GLESIndexBuffer(GLESBuffer):
    gltype = gl.GL_ELEMENT_ARRAY_BUFFER
    def __init__(self, data):
        super().__init__(data)

def calc_normals(points, mesh, face_normals):
    n = points.shape[0] // 6
    for i in range(n):
        # For vertex i, find all triangles incident and then average vertices
        faces = np.nonzero(mesh==i)[0] // 3
        ns = face_normals[faces]
        n = ns.sum(0)/faces.shape[0]
        n /= np.linalg.norm(n)
        points[i * 6 + 3:i*6+6] = n

class BranchMeshes(object):
    '''Class representing the branch meshes'''

    def __init__(self, n=32):
        rad_points = np.array(2 * np.pi * np.linspace(0, 1, n, dtype=np.float32))
        circle_x = np.cos(rad_points)
        circle_y = np.sin(rad_points)
        circle_z_0 = z = np.zeros_like(circle_x)
        circle_z_1 = circle_z_0 + 1

        circle = np.concatenate((np.stack((circle_x, circle_z_0, circle_y, z, z, z)).transpose().flatten(),
                                 np.stack((circle_x, circle_z_1, circle_y, z, z, z)).transpose().flatten()))

        def top(i):
            return n + (i % n)
        def bottom(i):
            return i % n

        mesh = np.array(
                [x
                 for i in range(n)
                 for x in (top(i), bottom(i), top(i + 1), top(i + 1), bottom(i), bottom(i+1))],
                dtype=np.uint16)

        mid_x = (circle_x + np.roll(circle_x,1))/2
        mid_y = (circle_y + np.roll(circle_y,1))/2
        d = mid_x * mid_x + mid_y * mid_y
        mid_x /= d
        mid_y /= d
        face_normals = np.stack((mid_x, np.zeros_like(mid_x), mid_y)).transpose() # Need to double them
        face_normals = np.stack((face_normals, face_normals)).transpose((1,0,2)).reshape(-1,3)

        calc_normals(circle, mesh, face_normals)

        self.cylinder_points = GLESVertexBuffer(circle)
        self.cylinder_mesh = GLESIndexBuffer(mesh)
        self.mesh_point_count = mesh.shape[0]

    def draw_branch(self):
        with self.cylinder_mesh, self.cylinder_points:
            gl.glDrawElements(gl.GL_TRIANGLES, self.mesh_point_count, gl.GL_UNSIGNED_SHORT, None)

def unitary_normal_basis(normal):
    n = glm.normalize(normal)
    if n.z < -0.9999999:
        t = glm.vec3(0, -1, 0)
        b = glm.vec3(-1, 0, 0)
    else:
        a = 1.0 / (1.0 + n.z)
        b2 = -n.x * n.y * a
        t = glm.vec3(1.0 - n.x * n.x * a,
                     b2,
                     -n.x)
        b = glm.vec3(b2, 1.0 - n.y * n.y * a, -n.y)
    return t, n, b

SIMPLE_TREE = [{ 'length': 10.0, 'bottom': 2.0, 'top': 0.9, 'point': glm.vec3(0, 1, 0)},
        [{ 'length': 3, 'bottom': 0.9, 'top': 0.4, 'point': glm.vec3(0.5, 3, 0.5)},
         {'length':1, 'bottom': 0.4, 'top': 0, 'point': glm.vec3(0.5, 4, 2)},
         {'length':3, 'bottom': 0.4, 'top': 0, 'point': glm.vec3(0.5, 3, 10)}],
        { 'length': 3, 'bottom': 0.9, 'top': 0, 'point': glm.vec3(0.3, 2, -0.8)}]
# Large oak tree description
OAK= [
    # Trunk
    { 'length': 10.0, 'bottom': 3.5, 'top': 2.2, 'point': glm.vec3(0, 1, 0) },

    # Major left limb
    [
        { 'length': 6.0, 'bottom': 2.2, 'top': 1.2, 'point': glm.vec3(-2.5, 2.5, 0) },
        # Sub-branches
        [
            { 'length': 3.5, 'bottom': 1.2, 'top': 0.6, 'point': glm.vec3(-3.5, 3.5, 1.0) },
            { 'length': 2.0, 'bottom': 0.6, 'top': 0.0, 'point': glm.vec3(-4.0, 4.0, 2.5) },
            { 'length': 1.8, 'bottom': 0.5, 'top': 0.0, 'point': glm.vec3(-4.5, 3.8, -1.0) }
        ],
        [
            { 'length': 3.0, 'bottom': 1.1, 'top': 0.5, 'point': glm.vec3(-2.0, 4.0, -1.5) },
            { 'length': 1.5, 'bottom': 0.5, 'top': 0.0, 'point': glm.vec3(-2.5, 5.0, -3.0) }
        ]
    ],
    # Major right limb
    [
        { 'length': 6.5, 'bottom': 2.1, 'top': 1.1, 'point': glm.vec3(2.8, 2.8, 0.2) },
        [
            { 'length': 3.2, 'bottom': 1.1, 'top': 0.5, 'point': glm.vec3(3.8, 4.0, 1.5) },
            { 'length': 1.7, 'bottom': 0.5, 'top': 0.0, 'point': glm.vec3(4.5, 5.0, 3.0) },
            { 'length': 1.5, 'bottom': 0.4, 'top': 0.0, 'point': glm.vec3(4.2, 4.5, 0.0) }
        ],
        [
            { 'length': 2.8, 'bottom': 1.0, 'top': 0.4, 'point': glm.vec3(2.0, 4.2, -1.8) },
            { 'length': 1.4, 'bottom': 0.4, 'top': 0.0, 'point': glm.vec3(2.5, 5.0, -3.5) }
        ]
    ],
    # Rear limb
    [
        { 'length': 5.5, 'bottom': 2.0, 'top': 1.0, 'point': glm.vec3(0.0, 2.3, -2.5) },
        [
            { 'length': 3.0, 'bottom': 1.0, 'top': 0.4, 'point': glm.vec3(0.5, 3.8, -4.0) },
            { 'length': 1.6, 'bottom': 0.4, 'top': 0.0, 'point': glm.vec3(1.2, 4.8, -6.0) }
        ],
        [
            { 'length': 2.5, 'bottom': 0.9, 'top': 0.3, 'point': glm.vec3(-1.0, 3.6, -3.8) },
            { 'length': 1.3, 'bottom': 0.3, 'top': 0.0, 'point': glm.vec3(-2.0, 4.5, -5.5) }
        ]
    ],
    # Upper canopy limb
    [
        { 'length': 4.5, 'bottom': 1.8, 'top': 0.8, 'point': glm.vec3(0.2, 4.5, 1.2) },
        [
            { 'length': 2.5, 'bottom': 0.8, 'top': 0.3, 'point': glm.vec3(1.2, 6.0, 2.5) },
            { 'length': 1.2, 'bottom': 0.3, 'top': 0.0, 'point': glm.vec3(2.0, 7.0, 4.0) }
        ],
        [
            { 'length': 2.2, 'bottom': 0.7, 'top': 0.3, 'point': glm.vec3(-1.0, 6.2, 1.8) },
            { 'length': 1.1, 'bottom': 0.3, 'top': 0.0, 'point': glm.vec3(-2.2, 7.0, 3.5) }
        ]
    ]
]

def generate_tree(r, level, size):
    length = (r.random() * 3 + 5) * ( 1 -  level / 15.0)
    top_size = r.random() * size

    theta = math.radians(45)
    z = math.cos(theta) * (1 - math.cos(theta)) * r.random()
    phi = 2 * math.pi * r.random()
    rad = math.sqrt(1 - z * z)

    if level == 0:
        point = glm.vec3(0,1,0)
    else:
        point = glm.vec3(rad * math.cos(phi), z, rad * math.sin(phi))

    print(point)
    info = {'length': length,
            'bottom': size, 'top': top_size,
            'point': point }

    if level > 8:
        return info
    else:
      if r.random() < top_size * 5.0 and top_size > (1/12.0):
          branch_count = int(r.random() * 4)
          return [info] + [generate_tree(r, level + 1, top_size) for b in range(branch_count)]
      else:
          return info

class DemoScene(object):
    def __init__(self, width, height):
        self.branch_vertex_shader = GLESShader('''
        attribute vec3 a_position;
        attribute vec3 a_normal;

        uniform mat4 u_projview; // model view projection matrix
        uniform mat4 u_model;
        uniform mat3 u_normalmat;
        uniform vec3 u_color;

        varying vec3 v_normal;
        varying vec3 v_worldPos;
        varying float v_radius01;
        varying vec2 v_uv;

        uniform vec4 u_params;

        const float PI = 3.14159265359;

        void main() {
          float u_bottomsize = u_params.x;
          float u_topsize = u_params.y;
          float u_trunkwidth = u_params.z;
          float h = u_params.w;

          float t = clamp(a_position.y, 0.0, 1.0);
          float r =  mix(u_bottomsize, u_topsize, t);

          vec4 pos = vec4(a_position.x * r, a_position.y * h, a_position.z * r, 1.0);

          vec4 worldPos = u_model * pos;
          v_worldPos = worldPos.xyz;
          v_normal = normalize(u_normalmat * a_normal);
          v_radius01 = mix(u_bottomsize / u_trunkwidth, u_topsize / u_trunkwidth, a_position.y);

          float ang = atan(a_position.z, a_position.x);
          float u = ang / (2.0 * PI) + 0.5;
          v_uv = vec2(u, a_position.y);
          gl_Position = u_projview * worldPos;
        }
        ''', gl.GL_VERTEX_SHADER)
        self.branch_fragment_shader = GLESShader(BRANCH_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
        self.global_illumination_shader = GLESShader('''
          precision mediump float;
          varying vec3 v_normal;
          uniform vec3 u_color;
          uniform vec3 u_lightDir;
          uniform float u_ambient;

          void main() {
            vec3 N = normalize(v_normal);
            vec3 L = normalize(u_lightDir);

            float diff = max(min(dot(N, L),1.0), 0.0);
            float brightness = u_ambient + (1.0 - u_ambient) * diff;
            gl_FragColor = vec4(u_color * brightness, 1.0);
          }
        ''', gl.GL_FRAGMENT_SHADER)
        self.sunlight_shader = GLESProgram('''
          attribute vec3 a_position;
          attribute vec3 a_normal;
          uniform mat4 u_projview;
          uniform mat4 u_model;
          uniform mat3 u_normalmat;
          varying vec3 v_normal;
          void main() {
            vec4 worldPos = u_model * vec4(a_position, 1.0);
            v_normal = normalize(u_normalmat * a_normal);
            gl_Position = u_projview * worldPos;
          }
        ''', fprog=self.global_illumination_shader)
        self.branch_shader = GLESProgram(self.branch_vertex_shader, fprog=self.branch_fragment_shader)
        self.meshes = BranchMeshes()
        self.position = glm.vec2(0, -10)

        self.proj = glm.perspective(
            glm.radians(60),
            width / float(height),
            0.1,
            100.0)

        self.tree = generate_tree(random.Random(random.random() * 10203141204), 0, 1)

        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.v_camera_angle_x = 0.0 # Radians to rotate the camera in the X direction every second
        self.v_camera_angle_y = 0.0
        self.vx = self.vy = 0
        self._last_time = None
        self._mouse_start = None

        self._bind_parameters(self.sunlight_shader)
        self._bind_parameters(self.branch_shader)
        self.reset_projview()
        self.reset_model_matrix()

        with self.branch_shader:
            gl.glUniform1f(self.branch_shader.uniforms.u_barkScale.loc, 0.25)
            gl.glUniform1f(self.branch_shader.uniforms.u_detailScale.loc, 2.0)
            gl.glUniform1f(self.branch_shader.uniforms.u_crackStrength.loc, 0.7)
            gl.glUniform1f(self.branch_shader.uniforms.u_normalStrength.loc, 0.35)

    def _bind_parameters(self, shader):
        vertex_stride = 6 * 4

        with shader:
            loc = shader.attribs.a_position.loc
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))

            gl.glEnableVertexAttribArray(shader.attribs.a_normal.loc)
            gl.glVertexAttribPointer(shader.attribs.a_normal.loc, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(12))

            light_dir = np.array([-0.3, 1.0, -0.2], dtype=np.float32)
            light_dir /= np.linalg.norm(light_dir)
            gl.glUniform3fv(shader.uniforms.u_lightDir.loc, 1, light_dir)
            gl.glUniform3fv(shader.uniforms.u_color.loc, 1, np.array([0.55,0.38,0.22], dtype=np.float32))
            gl.glUniform1f(shader.uniforms.u_ambient.loc,  0.2)

    def _accum_camera(self, elapsed):
        d_camera_angle_x = self.v_camera_angle_x * elapsed
        self.camera_angle_x = (self.camera_angle_x + d_camera_angle_x) % (2 * np.pi)

        d_camera_angle_y = self.v_camera_angle_y * elapsed
        self.camera_angle_y = max(-np.pi, min(np.pi, self.camera_angle_y + d_camera_angle_y))

        d_x = self.vx * elapsed * 3
        d_y = self.vy * elapsed * 3

        self.position += glm.vec2(d_x * math.sin(self.camera_angle_x) +
                                  d_y * math.cos(self.camera_angle_x),
                                  d_x * math.cos(self.camera_angle_x) -
                                  d_y * math.sin(self.camera_angle_x))

        if d_camera_angle_x != 0 or d_camera_angle_y != 0 or d_x != 0 or d_y != 0:
            self.reset_projview()

    def __call__(self, **kwargs):
        t = time.perf_counter()
        if self._last_time is not None:
            elapsed = t - self._last_time
            self._accum_camera(elapsed)

        self._last_time = t

        self._draw()

    def key_callback(self, win, glwin, key, scancode, action, mods):
        if key == glfw.KEY_Q:
            exit(0)
        if key in (glfw.KEY_UP,glfw.KEY_DOWN,glfw.KEY_LEFT,glfw.KEY_RIGHT):
            if action == glfw.PRESS:
                if key == glfw.KEY_UP:
                    self.vx, self.vy = 1, 0
                elif key == glfw.KEY_DOWN:
                    self.vx, self.vy = -1, 0
                elif key == glfw.KEY_LEFT:
                    self.vx, self.vy = 0, 1
                elif key == glfw.KEY_RIGHT:
                    self.vx, self.vy = 0, -1
            elif action == glfw.RELEASE:
                self.vx = self.vy = 0

    def mouse_pos_callback(self, win, glwin, xpos, ypos):
        if self._mouse_start is not None:
            sx, sy = self._mouse_start
            self.v_camera_angle_x = (min((xpos - sx) / 20.0, 30) / 360) * 2 * np.pi
            self.v_camera_angle_y = (min((ypos - sy) / 20.0, 60) / 360) * 2 * np.pi

    def mouse_button_callback(self, win, glwin, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._mouse_start = glfw.get_cursor_pos(glwin)
            else:
                self.v_camera_angle_x = 0
                self.v_camera_angle_y = 0
                self._mouse_start = None

    def reset_projview(self, model=None):
        viewer = glm.vec3(self.position.x, 6, self.position.y)
        look_at = glm.vec3(self.position.x + 10 * math.sin(self.camera_angle_x),
                           6 - 6 * math.sin(self.camera_angle_y),
                           self.position.y + 10 * math.cos(self.camera_angle_x) + 6 * math.cos(self.camera_angle_y))


        # Set a new view matrix based on the transformed camera position
        view = glm.lookAt(viewer, look_at, glm.vec3(0, 1, 0))

        # self.position is a glm.vec2 with x, y coords that ought to map to x, y coordinates that rerpesen camera position. View is set to wher 0,0 is now and we just need to apply a transform to it so that the right part of the 
        self.projview = self.proj * view

        self._reset_pvmat(self.sunlight_shader, self.projview)
        self._reset_pvmat(self.branch_shader, self.projview)

    def _reset_pvmat(self, shader, m):
        with shader:
            gl.glUniformMatrix4fv(shader.uniforms.u_projview.loc, 1, gl.GL_FALSE, glm.value_ptr(m))

    def reset_model_matrix(self, model=None):
        if model is None:
            model = glm.mat4(1.0)
        minv = glm.transpose(glm.inverse(glm.mat3(model)))
        self._reset_model_matrix(self.sunlight_shader, model, minv)
        self._reset_model_matrix(self.branch_shader, model, minv)

    def _reset_model_matrix(self, shader, m, minv):
        with shader:
            gl.glUniformMatrix4fv(shader.uniforms.u_model.loc, 1, gl.GL_FALSE, glm.value_ptr(m))
            gl.glUniformMatrix3fv(shader.uniforms.u_normalmat.loc, 1, gl.GL_FALSE, glm.value_ptr(minv))

    def _draw(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Length is the length of the limb (in feet), bottom is the width of the bottom of the branch (in feet), top is the width of the top of the branch. bottom is what hits the ground or the last branch. 'point' is a vector (not necessarily unit) that points in the direction the branch ought to point.
        # AI: using the above, generate a description of a large oak. Use the example of my silly tree below to see how to write it. Essentially a dictionary is a leaf branch. To include a branch with subbranches make a list. The first branch structure describes the root, then the remaining all branch from that root.
        # AI: keep tree and add 'oak'
        # AI: The oak should be large, and that means a lot more branches that make sense for an oak.

        tree = self.tree
        def draw_tree(t, m, trunk_width=None):
            if isinstance(t, dict):
                info = t
                branches = []
            else:
                info = t[0]
                branches = t[1:]

            point_dir = info.get('point') or glm.vec3(0, 1, 0)

            # Plane is point_dir.x * x + point_dir.y * y + point_dir.z * z = 0
            xvec, yvec, zvec = unitary_normal_basis(point_dir)
            rot = glm.mat4(glm.vec4(xvec, 0),
                           glm.vec4(yvec, 0),
                           glm.vec4(-zvec, 0),
                           glm.vec4(0, 0, 0, 1))
            m_branch = m * rot # Rotate first
            self.reset_model_matrix(m_branch)
            # Set normal matrix to inverse of m_branch

            length = info.get('length', 1.0)
            bottom = info.get('bottom', trunk_width or 1.0)
            top = info.get('top', trunk_width or 1.0)
            gl.glUniform4fv(self.branch_shader.uniforms.u_params.loc, 1, np.array([bottom, top, trunk_width or bottom, length]))
            self.meshes.draw_branch()

            neworigin = rot * glm.vec4(0,-length,0,1)
            m = m * glm.translate(glm.mat4(1.0), -neworigin.xyz)
            for b in branches:
                draw_tree(b, m, trunk_width = top)

        with self.branch_shader:
            draw_tree(tree, glm.mat4(1.0))

class Tree(object):
    def __init__(self, tree_model):
        self.model = tree_model

if __name__ == "__main__":
    def draw_triangle(*args, **kwargs):
        # Triangle at (0,0,0) (0,1,0) (1,1,0)
        pass

    win = EGLWindow(draw_triangle, "Test")
    with win:
        w, h = win.window_size()
        scene = DemoScene(w, h)
        win.renderfunc = scene
        win._listener = scene
        win._update_listener()
#        gl.glEnable(gl.GL_MULTISAMPLE)
#        win.listener = scene
    win.mainloop()
