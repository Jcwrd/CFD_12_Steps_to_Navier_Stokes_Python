import numpy
import numba
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# variable declarations
nx = 201
ny = 201
nt = 150
c = 2.5
dx = 4 / (nx - 1)
dy = 4 / (ny - 1)
sigma = .2
dt = sigma * dx

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

u = numpy.ones((ny, nx))
un = numpy.ones((ny, nx))
# Assign initial conditions

# set hat function
u[int(.25 / dy):int(1. / dy + 0.5), int(.25 / dx):int(1. / dx + 0.5)] = 2

# figure setting
fig, ax = plt.subplots(figsize=(7, 6))


im = ax.imshow(u, cmap='viridis', extent=[0, 4, 0, 4], origin='lower', vmin=1, vmax=2)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('u')

ax.set_title('2D Linear Convection Field')
ax.set_xlabel('X')
ax.set_ylabel('Y')


@numba.njit
def math_parth(u, un, c, dt, dx, nx, dy, ny):
    un = u.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
                       (c * dt / dy * (un[j, i] - un[j - 1, i])))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
    return u


# function of frame update
def animate(frame):
    global u
    if frame > 0:
        math_parth(u, un, c, dt, dx, nx, dy, ny)
        im.set_array(u)
    return im,


# calling animation
ani = FuncAnimation(fig, animate, frames=nt, interval=0, blit=False)
#ani.save("2D_linear_convection.gif", writer='pillow',fps=60)
plt.show()
