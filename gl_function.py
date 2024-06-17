from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import quaternion

def draw_axis(pos = np.array([0.0,0.0,0.0]), ori = np.quaternion(1.0, 0.0, 0.0, 0.0)):
    glPushMatrix()

    glTranslatef(pos[0], pos[1], pos[2])
    q = quaternion.as_rotation_vector(ori)
    glRotatef(np.rad2deg(np.linalg.norm(q)), q[0], q[1], q[2])

    glDisable(GL_LIGHTING)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)
    glEnd()

    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)
    glEnd()

    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 1.0)
    glEnd()
    glEnable(GL_LIGHTING)
    glPopMatrix()

def draw_cube(size):
    glScaled(size[0], size[1], size[2])

    n = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0]
    ])
    vn = np.array([
            [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
            [-1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0],
            [-1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [-1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0],
            [1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
            [1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0]
        ])
    faces = np.array([
        [0, 1, 2, 3],
        [3,2,6,7],
        [7,6,5,4],
        [4,5,1,0],
        [5,6,2,1],
        [7,4,0,3]
    ])

    v = np.zeros([8,3])


    v[0,0] = v[1,0] = v[2,0] = v[3,0] = -1.0/2
    v[4,0] = v[5,0] = v[6,0] = v[7,0] = 1.0/2
    v[0,1] = v[1,1] = v[4,1] = v[5,1] = -1.0/2
    v[2,1] = v[3,1] = v[6,1] = v[7,1] = 1.0/2
    v[0,2] = v[3,2] = v[4,2] = v[7,2] = -1.0/2
    v[1,2] = v[2,2] = v[5,2] = v[6,2] = 1.0/2
    
    for i in range(5,-1,-1):
        glBegin(GL_QUADS)
        glNormal3fv(n[i])
        for j in range(4):
            glVertex3fv(v[faces[i,j]])
        glEnd()


