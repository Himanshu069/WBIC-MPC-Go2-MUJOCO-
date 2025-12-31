import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math

xml_path = 'unitree_go2/scene.xml' #xml file (assumes this is in the same folder as this file)
simend = 50 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

import threading
import tkinter as tk
import numpy as np

# Shared joystick commands
joystick_state = {
    "x": 0.0,  # left/right
    "y": 0.0   # forward/back
}

def joystick_window():
    def on_drag(event):
        # Normalize mouse coordinates to range [-1, 1]
        x = (event.x - canvas_size/2) / (canvas_size/2)
        y = -(event.y - canvas_size/2) / (canvas_size/2)
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        joystick_state["x"] = x
        joystick_state["y"] = y
        # Update visual indicator
        canvas.coords(dot, canvas_size/2 + x*(canvas_size/2), canvas_size/2 - y*(canvas_size/2),
                      canvas_size/2 + x*(canvas_size/2) + 10, canvas_size/2 - y*(canvas_size/2) + 10)

    root = tk.Tk()
    root.title("Virtual Joystick")
    global canvas_size
    canvas_size = 200
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='lightgray')
    canvas.pack()
    dot = canvas.create_oval(canvas_size/2-5, canvas_size/2-5, canvas_size/2+5, canvas_size/2+5, fill='red')
    canvas.bind("<B1-Motion>", on_drag)  # drag with left button
    root.mainloop()

# Start joystick in a separate thread
threading.Thread(target=joystick_window, daemon=True).start()


def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    data.ctrl[:] = 0.0    

def controller(model, data):
    forward = joystick_state["y"]  # -1 to 1
    turn = joystick_state["x"]     # -1 to 1

    # Apply to motor control
    # This depends on how your robot joints are mapped
    data.ctrl[:] = forward 
    
    
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = -91.79999999999993 ; cam.elevation = -39.600000000000016 ; cam.distance =  0.7590817024851124
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    # if data.qpos[1]>6.28:
    #     angle = data.qpos[1] - math.floor(data.qpos[1]/6.28)*6.28
    # else:
    #     angle = data.qpos[1]

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()

