from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
from random import random
from numpy import array, zeros, sqrt, meshgrid
from scipy.integrate import odeint

name = 'nbody'
univ = None
camera_xyz = [0.,0.,100.,]
W=400
H=400
trace_len=100
pause = False
DT = 1

def deriv(q, t0, m):
    n = q.shape[0] / 6
    d = zeros(q.shape)
    d[:3 * n] = q[3 * n:]
    x = q[0:3 * n:3]
    y = q[1:3 * n:3]
    z = q[2:3 * n:3]
    x1,x2 = meshgrid(x,x)
    y1,y2 = meshgrid(y,y)
    z1,z2 = meshgrid(z,z)
    d2 = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    m1,m2 = meshgrid(m,m)
    x21 = x2-x1
    y21 = y2-y1
    z21 = z2-z1
    for i in xrange(d2.shape[0]):
        d2[i,i] = 1
    fx = (x21 * m2 / d2 ** (3. / 2)).sum(0)
    fy = (y21 * m2 / d2 ** (3. / 2)).sum(0)
    fz = (z21 * m2 / d2 ** (3. / 2)).sum(0)
    d[3 * n::3] = fx
    d[3 * n + 1::3] = fy
    d[3 * n + 2::3] = fz
    return d

class universe:
    def __init__(self, bodies, dt=1.):
        self.dt = dt
        self.bodies = bodies
        self.n = len(bodies)
        self.q = zeros(6 * self.n)
        self.q[:3 * self.n] = gather_vec(bodies, "xyz")
        self.q[3 * self.n:] = gather_vec(bodies, "vxyz")
        self.m = array(map(lambda b: b.m, bodies))

    def update(self):
        self.q = odeint(deriv, self.q, [0,self.dt], (self.m,))[1]
        x = self.q[0:3 * self.n].reshape(self.n, 3)
        M = self.m.sum()
        x -= (x.transpose() * self.m).sum(1) / M
        scatter_vec(self.q[:3 * self.n], self.bodies, "xyz")
        for b in self.bodies:
            b.trace.append(b.xyz)
            while len(b.trace) > trace_len:
                b.trace.pop(0)


class body:
    def __init__(self,
                 xyz,
                 r = 1.,
                 rgb = [1.,1.,1.],
                 m = 1,
                 vxyz =[0.,0.,0.],
                 ergb =[0.,0.,0.],
                 ls = 0):
        self.xyz = xyz
        self.r = r
        self.rgb = rgb
        self.ergb = ergb
        self.m = m
        self.vxyz = vxyz
        self.collided = -1
        self.ls = ls
        self.trace = []
        self.trace.append(self.xyz)

def gather_vec(bodies, prop):
    return reduce(lambda a,b: a + b, map(lambda b: b.__dict__[prop], bodies))

def scatter_vec(x, bodies, prop):
    for i in xrange(len(bodies)):
        bodies[i].__dict__[prop] = list(x[3 * i : 3 * (i+1)])

def collide(univ):
    coll = 0
    n = len(univ.bodies)
    bodies = univ.bodies
    for i in xrange(n - 1):
        b1 = bodies[i]
        if b1.collided >= 0:
            continue
        xyz1 = array(b1.xyz)
        vxyz1 = array(b1.vxyz)
        for j in xrange(i+1,n):
            b2 = bodies[j]
            xyz2 = array(b2.xyz)
            if sqrt(((xyz2 - xyz1)**2).sum()) < b1.r + b2.r:
                coll += 1
                if b2.collided >= 0:
                    b2 = bodies[b2.collided]
                    xyz2 = array(b2.xyz)
                vxyz2 = array(b2.vxyz)
                b2.collided = i
                b1.xyz = list((b1.m * xyz1 + b2.m * xyz2) / (b1.m + b2.m))
                b1.vxyz = list((b1.m * vxyz1 + b2.m * vxyz2) / (b1.m + b2.m))
                b1.m = b1.m + b2.m
                b1.r = (b1.r**3 + b2.r**3) ** (1./3)
    if coll == 0:
        return univ
    newbodies = []
    for b in bodies:
        if b.collided < 0:
            newbodies.append(b)
    return universe(newbodies)

def setup_univ():
    dt = 2
    bodies = []
    M = 10.
    bodies.append(body( [0.,0.,0.],
                        r=5,
                        ergb=[1.,1.,0.],
                        m=M,
                        ls = 1,
                        ))
    R = 30
    bodies.append(body( [R,0.,0.],
                        r=5,
                        rgb=[1.,0.,0.],
                        m=1,
                        vxyz=[0.,sqrt(M/R),0.],
                      ))
    R = 40
    bodies.append(body( [0.,0.,R],
                        r=5,
                        rgb=[0.,1.,0.],
                        m=1,
                        vxyz=[0.,-sqrt(M/R),0.],
                       ))
    return universe(bodies, dt)

def main():
    global univ

    univ = setup_univ()
    
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(W,H)
    glutCreateWindow(name)

    glClearColor(.05,0.,.1,1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    setup_camera()

    glLightfv(GL_LIGHT0, GL_POSITION, [0.,0.,0.,1.])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.,1.,1.,1.])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.,0.,0.,1.])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.,0.,0.,0.])
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.5)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0)
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0)
    glEnable(GL_LIGHT0)

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)

    glutMainLoop()
    return

def display():
    global univ

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    for b in univ.bodies:
        if b.ls:        
            glLightfv(GL_LIGHT0, GL_POSITION, b.xyz + [1.])
        glPushMatrix()
        glTranslatef(*b.xyz)
        glMaterialfv(GL_FRONT,GL_DIFFUSE,b.rgb + [1.])
        glMaterialfv(GL_FRONT,GL_EMISSION,b.ergb + [1.])
        glutSolidSphere(b.r,50,50)

        glPopMatrix()

        glMaterialfv(GL_FRONT,GL_EMISSION,b.rgb + [1.])
        glBegin(GL_LINE_STRIP)
        for p in b.trace:
            glVertex3f(*p)
        glEnd()


    glutSwapBuffers()

    if not pause:
        univ.update()

    glutPostRedisplay();
    return

def reshape(w,h):
    global W,H

    W,H = w,h
    glViewport(0,0,w,h)
    setup_camera()
    glutPostRedisplay();

def keyboard(c, x, y):
    global camera_xyz, trace_len, pause

    if c == "+":
        univ.dt *= 1.5
    elif c == "-":
        univ.dt /= 1.5
    elif c == "q":
        glutLeaveMainLoop()
    elif c == 's' or c == 'w':
        v = array(camera_xyz)
        if c == 's':
            camera_xyz = list(1.05 * v)
        else:
            camera_xyz = list(v / 1.05)
        setup_camera()
    elif c == 't':
            trace_len -= 10
            if trace_len < 0:
                trace_len = 0
    elif c == 'T':
        trace_len += 10
    elif c in ' ':
        pause = not pause

def setup_camera():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90.,float(W)/H,1.,2 * sqrt((array(camera_xyz)**2).sum()))
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(camera_xyz[0],camera_xyz[1],camera_xyz[2],
              0,0,0,
              0,1,0)

if __name__ == '__main__': main()
