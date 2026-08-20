import numpy                 # we're importing numpy
import sympy
from matplotlib import pyplot, pyplot as plt  # and our 2D plotting library
from matplotlib.animation import FuncAnimation, PillowWriter

from sympy import lambdify
x,nu,t=sympy.symbols('x nu t')
phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
       sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
phiprime=phi.diff(x)
u=-2*nu*(phiprime/phi)+4
print(u)
ufunc = lambdify((t, x, nu), u)


nx=201
nt=100
dx=2*numpy.pi/(nx-1)
nu=0.01
dt=dx*nu
x=numpy.linspace(0,2*numpy.pi,nx)
un=numpy.empty(nx)
t=0

u = numpy.asarray([ufunc(t, x0, nu) for x0 in x])


fig,(ax1,ax2)=pyplot.subplots(2,1,figsize=(15,7),gridspec_kw={'height_ratios':[5,1]})
x_coords=numpy.linspace(0,2*numpy.pi,nx)
line,=ax1.plot(x_coords,u,color='blue',lw=2)
ax1.set_title('diffusion equation')
ax1.set_xlabel('x')
ax1.set_ylabel('u')
ax1.grid(True)

pipe_data=numpy.atleast_2d(u)
im=ax2.imshow(pipe_data,cmap='viridis',extent=[0,2*numpy.pi,0,1],vmin=1,vmax=7)

cbar=fig.colorbar(im, ax=[ax1,ax2], orientation='vertical',pad=0.05,shrink=1)

def animate(frame):
    global u
    if frame==0:
        global u
        u = numpy.asarray([ufunc(t, x0, nu) for x0 in x])
    else:
        for n in range(nt):
            un = u.copy()
            for i in range(1, nx - 1):
                u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1]) + nu * dt / dx ** 2 * \
                       (un[i + 1] - 2 * un[i] + un[i - 1])
            u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx ** 2 * \
(un[1] - 2 * un[0] + un[-2])
            u[-1] = u[0]
    line.set_data(x_coords,u)
    im.set_array(numpy.atleast_2d(u))
    return im,line
ani=FuncAnimation(fig, animate, frames=500,interval=10,blit=True)
#ani.save("Burgers equation.gif", writer='pillow',fps=30)
plt.show()
