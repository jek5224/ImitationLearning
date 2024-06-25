import imgui
import glfw

from OpenGL.GL import *
from OpenGL.GLU import *

from imgui.integrations.glfw import GlfwRenderer

import numpy as np
import quaternion

from env import Env as MyEnv
from TrackBall import TrackBall
import gl_function as mygl
from ray_model import loading_network

import dartpy as dart

from PIL import Image
from PIL import ImageOps

width = 1920
height = 1080

def impl_glfw_init(window_name="Imitation Learning", width=width, height=height):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize window")
        exit(1)

    return window

class GLFWApp():
    def __init__(self):
        super().__init__()

        self.name = "Imitation Learning"
        self.width = 1920
        self.height = 1080

        self.perspective = 45.0
        self.eye = np.array([0.0, 0.0, 1.0])    # Position of camera
        self.up = np.array([0.0, 1.0, 0.0])
        self.trans = np.array([0.0, 0.0, 0.0])
        self.zoom = 1.0

        self.trackball = TrackBall()
        self.trackball.set_trackball(np.array([self.width * 0.5, self.height * 0.5]), self.width * 0.5)
        self.trackball.set_quaternion(np.quaternion(1.0, 0.0, 0.0, 0.0))

        self.mouse_down = False
        self.rotate = False
        self.translate = False

        self.mouse_x = 0
        self.mouse_y = 0

        self.is_simulation = False
        self.draw_object = False

        self.motion_skel = None
        
        # For Screenshot
        self.imagenum = 0

        self.reset_value = 0

        imgui.create_context()
        self.window = impl_glfw_init(self.name, self.width, self.height)
        self.impl = GlfwRenderer(self.window)

        def framebuffer_size_callback(window, width, height):
            self.width = width
            self.height = height
            glViewport(0, 0, width, height)

        glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)

        def mouseButtonCallback(window, button, action, mods):
            if not imgui.get_io().want_capture_mouse:
                self.mousePress(button, action, mods)
        
        glfw.set_mouse_button_callback(self.window, mouseButtonCallback)

        def cursorPosCallBack(window, xpos, ypos):
            if not imgui.get_io().want_capture_mouse:
                self.mouseMove(xpos, ypos)

        glfw.set_cursor_pos_callback(self.window, cursorPosCallBack)

        def scrollCallBack(window, xoffset, yoffset):
            if not imgui.get_io().want_capture_mouse:
                self.mouseScroll(xoffset, yoffset)

        glfw.set_scroll_callback(self.window, scrollCallBack)

        def keyCallBack(window, key, scancode, action, mods):
            if not imgui.get_io().want_capture_mouse:
                self.keyboardPress(key, scancode, action, mods)

        glfw.set_key_callback(self.window, keyCallBack)

        self.env = None
        self.nn = None

        self.reward_buffer = []
    def mousePress(self, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_down = True
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.rotate = True
                self.trackball.start_ball(self.mouse_x, self.height - self.mouse_y)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = True
        elif action == glfw.RELEASE:
            self.mouse_down = False
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.rotate = False
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = False

    def mouseMove(self, xpos, ypos):
        dx = xpos - self.mouse_x
        dy = ypos - self.mouse_y

        self.mouse_x = xpos
        self.mouse_y = ypos

        if self.rotate:
            if dx != 0 or dy != 0:
                self.trackball.update_ball(xpos, self.height - ypos)

        if self.translate:
            rot = quaternion.as_rotation_matrix(self.trackball.curr_quat)
            self.trans += (1.0 / self.zoom) * rot.transpose() @ np.array([dx, -dy, 0.0])

    def mouseScroll(self, xoffset, yoffset):
        if yoffset < 0:
            self.eye *= 1.05
        elif yoffset > 0 and np.linalg.norm(self.eye) > 0.5:
            self.eye *= 0.95
    
    def keyboardPress(self, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_SPACE:
                self.is_simulation = not self.is_simulation
            elif key == glfw.KEY_S:
                self.update()
            elif key == glfw.KEY_R:
                self.reset(self.reset_value)

    def setEnv(self, env):
        self.env = env
        self.motion_skel = self.env.skel.clone()
        self.reset()

    def reset(self, reset_time=None):
        self.env.reset(reset_time)
        self.reward_buffer = [self.env.get_reward()]
    
    def loadNetwork(self, path):
        self.nn = loading_network(path, self.env.num_bobs, self.env.num_action)

    def startLoop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            self.impl.process_inputs()
            if self.is_simulation:
                self.update()

                # glPixelStorei(GL_PACK_ALIGNMENT, 1)
                # data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
                # image = Image.frombytes("RGBA", (width, height), data)
                # image = ImageOps.flip(image)
                # image.save(f'{self.imagenum}.png', 'PNG')
                # self.imagenum += 1

            self.drawSimFrame()
            self.drawUIFrame()

            self.impl.render(imgui.get_draw_data())
            
            glfw.swap_buffers(self.window)

        self.impl.shutdown()
        glfw.terminate()
        return

    def update(self):
        if self.nn is not None:
            obs = self.env.get_obs()
            action = self.nn.get_action(obs)
            _, _, done, _ = self.env.step(action)
        else:
            _, _, done, _ = self.env.step(np.array(np.zeros(self.env.skel.getNumDofs() - self.env.skel.getJoint(0).getNumDofs())))

        # if done:
        #     self.is_simulation = False
        
        self.reward_buffer.append(self.env.get_reward())

    def initGL(self):
        ## Light Option 
        ambient = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
        diffuse = np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32)
        front_mat_shininess = np.array([60.0], dtype=np.float32)
        front_mat_specular = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
        front_mat_diffuse = np.array([0.5, 0.28, 0.38, 1.0], dtype=np.float32)
        lmodel_ambient = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
        lmodel_twoside = np.array([GL_FALSE])
        light_pos = [    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
                        np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32)]
        
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glShadeModel(GL_SMOOTH)
        glPolygonMode(GL_FRONT, GL_FILL)

        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos[0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient)
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside)

        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT1, GL_POSITION, light_pos[1])

        glEnable(GL_LIGHT2)
        glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT2, GL_POSITION, light_pos[2])

        glEnable(GL_LIGHTING)

        glEnable(GL_COLOR_MATERIAL)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_MULTISAMPLE)

    def drawSimFrame(self):
        self.initGL()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glViewport(0, 0, self.width, self.height)
        gluPerspective(self.perspective, (self.width / self.height), 0.1, 100.0)
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], 0.0, 0.0, -1.0, self.up[0], self.up[1], self.up[2])
        

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.trackball.set_center(np.array([self.width * 0.5, self.height * 0.5]))
        self.trackball.set_radius(min(self.width, self.height) * 0.4)
        self.trackball.apply_gl_rotation()

        glScalef(self.zoom, self.zoom, self.zoom)
        glTranslatef(self.trans[0] * 0.001, self.trans[1] * 0.001, self.trans[2] * 0.001)
        glEnable(GL_DEPTH_TEST)

        self.drawGround(-1E-3)

        if self.mouse_down:
            mygl.draw_axis()

        self.drawSkeleton(self.env.skel.getPositions())
        self.drawSkeleton(self.env.target_pos, np.array([1.0, 0.3, 0.3, 0.5]))

    
    def drawGround(self, height):
        glDisable(GL_LIGHTING)
        w = 0.005
        count = 0

        for x in range(-20, 20):
            for z in range(-21, 20):
                if count % 2 == 0:
                    glColor3f(0.8, 0.8, 0.8)
                else:
                    glColor3f(0.7, 0.7, 0.7)

                glBegin(GL_QUADS)
                glVertex3f(x, height, z)
                glVertex3f(x + 1, height, z)
                glVertex3f(x + 1, height, z + 1)
                glVertex3f(x, height, z + 1)
                glEnd()

                count += 1

        glEnable(GL_LIGHTING)

    def drawSkeleton(self, pos, color=np.array([0.5, 0.5, 0.5, 1.0])):
        self.motion_skel.setPositions(pos)

        for bn in self.motion_skel.getBodyNodes():
            glPushMatrix()
            glMultMatrixd(bn.getWorldTransform().matrix().transpose())
            for sn in bn.getShapeNodes():
                if not sn:
                    return
                
                va = sn.getVisualAspect()

                if not va or va.isHidden():
                    return
                
                glPushMatrix()
                glMultMatrixd(sn.getRelativeTransform().matrix().transpose())
                self.drawShape(sn.getShape(), color)
                glPopMatrix()

            glPopMatrix()

    def drawShape(self, shape, color):
        if not shape:
            return
        
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glColor4d(color[0], color[1], color[2], color[3])

        if not self.draw_object:
            if type(shape) == dart.dynamics.BoxShape:
                mygl.draw_cube(shape.getSize())

    def drawUIFrame(self):
        imgui.new_frame()

        imgui.set_next_window_size(400, 400, condition=imgui.ONCE)
        imgui.set_next_window_position(self.width - 410, 10, condition=imgui.ONCE)

        imgui.begin("Information")
        imgui.text("Elapsed\tTime\t:\t%.2f" % self.env.world.getTime())


        if imgui.tree_node("Observation"):
            imgui.plot_histogram(
                label="##obs",
                values=self.env.get_obs().astype(np.float32),
                values_count=self.env.num_obs,
                scale_min=-10.0,
                scale_max= 10.0,
                graph_size=(imgui.get_content_region_available_width(), 200)
            )
            imgui.tree_pop()

        if imgui.tree_node("Reward"):
            width = 60
            data_width = min(width, len(self.reward_buffer))
            value = np.zeros(width, dtype=np.float32)
            value[-data_width:] = np.array(self.reward_buffer[-data_width:], dtype=np.float32)
            imgui.plot_lines(
                label="##reward",
                values=value,
                values_count=width,
                scale_min=0.0,
                scale_max=1.0,
                graph_size=(imgui.get_content_region_available_width(), 200)
            )
            imgui.tree_pop()

        changed, self.reset_value = imgui.slider_float("Reset Time", self.reset_value, 0.0, self.env.bvhs[self.env.bvh_idx].bvh_time)
        if imgui.button("Reset"):
            self.reset(self.reset_value)
        imgui.same_line()
        if imgui.button("Random Reset"):
            self.reset()
        

        imgui.end()
        imgui.render()

import argparse

parser = argparse.ArgumentParser(description='Imitation Learning')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint_path')

if __name__ == "__main__":
    args = parser.parse_args()
    
    app = GLFWApp()
    app.setEnv(MyEnv())
    if args.checkpoint:
        app.loadNetwork(args.checkpoint)

    app.startLoop()