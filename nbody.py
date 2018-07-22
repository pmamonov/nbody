from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
from random import random
from numpy import array, zeros, sqrt, meshgrid, arange
from scipy.integrate import odeint
from math import acos, cos, sin, pi

name = 'nbody'
univ = None
camera_xyz = [0.,0.,100.,]
W=400
H=400
trace_res = 5
trace_len = 100 * trace_res
pause = False
RMAX = 50
DT = 1
r_planet = 0
r_star = 0.5

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
        print self.q
        sys.stdout.flush()
        self.m = array(map(lambda b: b.m, bodies))

    def update(self):
        t = arange(0,self.dt,float(self.dt)/(trace_res + 1))
        r = odeint(deriv, self.q, t, (self.m,))
        r = r.reshape(t.shape[0],self.n * 2,3)
        x = 1. * r[:,:self.n,:]
        for i in xrange(t.shape[0]):
            x[i,:,:].transpose()[:,:] *= self.m
        M = self.m.sum()
        c = x.sum(1) / M
        for i in xrange(t.shape[0]):
            r[i,:self.n,:] -= c[i,:]
        self.q = r[-1,:,:].reshape(self.n*6)
        scatter_vec(self.q[:3 * self.n], self.bodies, "xyz")
        for i in xrange(self.n):
            b = self.bodies[i]
            b.trace += map(lambda n: list(r[n,i,:]), range(1,r.shape[0]))
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
    dt = .1
    bodies = []
    M = 1000.
    # a star
    bodies.append(body( [0.,0.,0.], # 1st arg is the list of coords
                        r=r_star, # spere radius
                        ergb=[1.,1.,0.], # RGB for material emission color (optional)
                        m=M, # mass (gravitational as well as inertial)
                        ls = 1, # light source anchor (optional)
                        ))

    # planets
    R = 30.
    for i in xrange(10):
        rgb = [.5+.5*random(),.5+.5*random(),.5+.5*random()]
        xyz = [R * random(), R*random(), .01* R *random()]
        r = sqrt((array(xyz)**2).sum())
        V = sqrt(M/r)
        vx = V * (2 * random() - 1)
        vy = sqrt(V**2 - vx**2)
        vxyz = [vx,vy,0.]
        bodies.append(body( xyz,
                            r=r_planet,
                            rgb=rgb, # RGB for material color
                            m=1,
                            vxyz=vxyz, # list of velocity components
                            ))
        M += bodies[-1].m
        R += 10
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
        
        if b.r > 0:
            glPushMatrix()
            glTranslatef(*b.xyz)
            glMaterialfv(GL_FRONT,GL_DIFFUSE,b.rgb + [1.])
            glMaterialfv(GL_FRONT,GL_EMISSION,b.ergb + [1.])
            glutSolidSphere(b.r,50,50)
            glPopMatrix()

        glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.,0.,0.,1.])
        glBegin(GL_LINE_STRIP)
        for i in xrange(0,len(b.trace),1):
            p = b.trace[-i - 1]
            rgb = map(lambda x: x * (trace_len - i) / trace_len, b.rgb)
            glMaterialfv(GL_FRONT,GL_EMISSION,rgb + [1.])
            glVertex3f(*p)
        glEnd()


    glutSwapBuffers()

    if not pause:
        univ.update()
        xyz = array(gather_vec(univ.bodies, "xyz"))
        R = (sqrt((xyz.reshape(univ.n, 3) ** 2).sum(1)) * univ.m).sum() / univ.m.sum()
#        print "%.2f" % R
        if R > RMAX:
            univ = setup_univ()

    glutPostRedisplay();
    return

def reshape(w,h):
    global W,H

    W,H = w,h
    glViewport(0,0,w,h)
    setup_camera()
    glutPostRedisplay();

def keyboard(c, x, y):
    global univ, camera_xyz, trace_len, pause

    if c == "+":
        univ.dt *= 1.5
    elif c == "-":
        univ.dt /= 1.5
    elif c == "q":
        glutLeaveMainLoop()
    elif c in 'awsdAWSD':
        x,y,z = camera_xyz
        r = sqrt(x**2+y**2+z**2)
        rxz = sqrt(x**2+z**2)
        theta = acos(y/r)
        phi = acos(x/rxz)
        if z < 0:
            phi *= -1
        if c == 's':
            r *= 1.05
        elif c == 'w':
            r /= 1.05
        elif c in 'aA':
            phi += 5 * pi / 180
        elif c in 'dD':
            phi -= 5 * pi / 180
        elif c == 'W':
            theta -= 5 * pi / 180
            if theta <= 0:
                theta = pi * 1e-3
        elif c == 'S':
            theta += 5 * pi / 180
            if theta >= pi:
                theta = pi * (1-1e-3)
        y = r * cos(theta)
        x = r * sin(theta) * cos(phi)
        z = r * sin(theta) * sin(phi)
        camera_xyz = [x,y,z]
        setup_camera()

    elif c == 't':
            trace_len -= 10
            if trace_len < 0:
                trace_len = 0
    elif c == 'T':
        trace_len += 10
    elif c in ' ':
        pause = not pause
    elif c == 'r':
        univ = setup_univ()

def setup_camera():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90.,float(W)/H,1.,-1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(camera_xyz[0],camera_xyz[1],camera_xyz[2],
              0,0,0,
              0,1,0)

if __name__ == '__main__': main()
