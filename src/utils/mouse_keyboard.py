import mujoco as mj
import glfw


class MouseKeyboard:
    def __init__(self, model, data, scene, cam):
        self.model = model
        self.data = data
        self.scene = scene
        self.cam = cam

        # mouse state
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

    # ---------- keyboard ----------
    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    # ---------- mouse button ----------
    def mouse_button(self, window, button, act, mods):
        self.button_left = ( glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = ( glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = ( glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS )
        glfw.get_cursor_pos(window)

    # ---------- mouse move ----------
    def mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        if not (self.button_left or self.button_middle or self.button_right):
            return

        width, height = glfw.get_window_size(window)
        mod_shift = ( glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if self.button_right:
            action = ( mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V)
        elif self.button_left:
            action = ( mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift
                else mj.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model,action,dx / height,dy / height, self.scene,self.cam, )

    # ---------- scroll ----------
    def scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0,-0.05 * yoffset,self.scene, self.cam,  )