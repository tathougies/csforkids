import ctypes
import ctypes.util
import tkinter

from OpenGL import EGL, setPlatform
from OpenGL import GLES3

# --- Xlib just to get VisualID ---
libX11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
Display_p = ctypes.c_void_p
Window = ctypes.c_ulong
Visual_p = ctypes.c_void_p

class XWindowAttributes(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
                ("visual", Visual_p),
                ("root", Window),
                ("class", ctypes.c_int),
                ("bit_gravity", ctypes.c_int),
                ("win_gravity", ctypes.c_int),
                ("backing_store", ctypes.c_int),
                ("backing_planes", ctypes.c_ulong),
                ("backing_pixel", ctypes.c_ulong),
                ("save_under", ctypes.c_int),
                ("colormap", ctypes.c_ulong),
                ("map_installed", ctypes.c_int),
                ("map_state", ctypes.c_int),
                ("all_event_masks", ctypes.c_long),
                ("your_event_mask", ctypes.c_long),
                ("do_not_propagate_mask", ctypes.c_long),
                ("override_redirect", ctypes.c_int),
                ("screen", ctypes.c_void_p)]

libX11.XOpenDisplay.argtypes = [ctypes.c_char_p]
libX11.XOpenDisplay.restype = Display_p
libX11.XGetWindowAttributes.argtypes = [Display_p, Window, ctypes.POINTER(XWindowAttributes)]
libX11.XGetWindowAttributes.restype = ctypes.c_int
libX11.XVisualIDFromVisual.argtypes = [Visual_p]
libX11.XVisualIDFromVisual.restype = ctypes.c_ulong

def get_visualid_for_xwindow(xwin: int) -> tuple[int, int]:
    dpy = libX11.XOpenDisplay(None)
    if not dpy:
        raise RuntimeError("XOpenDisplay failed (are you on X11/XWayland?)")
    xwa = XWindowAttributes()
    if libX11.XGetWindowAttributes(dpy, Window(xwin), ctypes.byref(xwa)) == 0:
        raise RuntimeError("XGetWindowAttributes failed")
    visualid = int(libX11.XVisualIDFromVisual(xwa.visual))
    return int(dpy), visualid  # dpy pointer value + visualid

class EGLCanvas(tkinter.Frame):
    @property
    def gl(self):
        return GLES3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setPlatform('egl')
        self.cont = None
        self.egl_dpy = None
        super().bind("<Map>", self._init_egl)
        super().bind("<Expose>", self.draw)

    def make_current(self):
        if self.egl_dpy is None:
            return
        ok = EGL.eglMakeCurrent(self.egl_dpy, self.surface, self.surface, self.egl_context)
        if not ok:
            raise RuntimeError(f"eglMakeCurrent failed: 0x{EGL.eglGetError():04x}")

    def swap_buffers(self):
        if self.egl_dpy is None:
            return
        EGL.eglSwapBuffers(self.egl_dpy, self.surface)

    def draw(self, *args):
        if self.egl_dpy is None:
            return
        self.make_current()
        if self.cont is not None:
            self.cont.send(None)

    def _init_egl(self, e):
        self.update()
        xwin = self.winfo_id()

        dpy_ptr, visualid = get_visualid_for_xwindow(xwin)

        self.egl_dpy = egl_dpy = EGL.eglGetDisplay(dpy_ptr)
        if egl_dpy == EGL.EGL_NO_DISPLAY:
            raise RuntimeError("eglGetDisplay failed")

        major, minor = ctypes.c_int(), ctypes.c_int()
        if not EGL.eglInitialize(egl_dpy, major, minor):
            raise RuntimeError("eglInitialize failed")

        EGL.eglBindAPI(EGL.EGL_OPENGL_ES_API)

        attribs = [
            EGL.EGL_SURFACE_TYPE, EGL.EGL_WINDOW_BIT,
            EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_ES2_BIT,
            EGL.EGL_RED_SIZE, 8,
            EGL.EGL_GREEN_SIZE, 8,
            EGL.EGL_BLUE_SIZE, 8,
            EGL.EGL_ALPHA_SIZE, 8,
            EGL.EGL_NONE
        ]

        num = ctypes.c_int()
        # get configs
        configs = (EGL.EGLConfig * 64)()
        if not EGL.eglChooseConfig(egl_dpy, attribs, configs, 64, num):
            raise RuntimeError("eglChooseConfig failed")

        chosen = None
        for i in range(num.value):
            vid = ctypes.c_int()
            EGL.eglGetConfigAttrib(egl_dpy, configs[i], EGL.EGL_NATIVE_VISUAL_ID, vid)
            if vid.value == visualid:
                chosen = configs[i]
                break
        if chosen is None:
            raise RuntimeError("No EGLConfig matches Tk window VisualID")

        self.surface = surface = EGL.eglCreateWindowSurface(egl_dpy, chosen, xwin, None)
        if surface == EGL.EGL_NO_SURFACE:
            raise RuntimeError("eglCreateWindowSurface failed")

        ctx_attribs = [EGL.EGL_CONTEXT_CLIENT_VERSION, 2, EGL.EGL_NONE]
        self.egl_context = ctx = EGL.eglCreateContext(egl_dpy, chosen, EGL.EGL_NO_CONTEXT, ctx_attribs)
        if ctx == EGL.EGL_NO_CONTEXT:
            raise RuntimeError("eglCreateContext failed")

        if not EGL.eglMakeCurrent(egl_dpy, surface, surface, ctx):
            raise RuntimeError("eglMakeCurrent failed")

        self.after(0, self.draw)
