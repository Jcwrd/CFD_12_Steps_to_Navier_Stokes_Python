import numpy
import numba
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# variable declarations
nx = 71
ny = 71
nt = 51
c = 2.5
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
v=0.05
sigma = .25
dt = sigma * dx*dy/v

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

u = numpy.ones((ny, nx))
un = numpy.ones((ny, nx))
# Assign initial conditions

# set hat function
u[int(.5 / dy):int(1. / dy + 1), int(.5 / dx):int(1. / dx + 1)] = 2

# figure setting
fig, ax = plt.subplots(figsize=(7, 6))


im = ax.imshow(u, cmap='viridis', extent=[0, 2, 0, 2], origin='lower', vmin=1, vmax=2)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('u')

ax.set_title('2D Linear Convection Field')
ax.set_xlabel('X')
ax.set_ylabel('Y')


@numba.njit
def math_parth(u, un, c, dt, dx, nx, dy, ny, v):
    un = u.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] = (un[j, i] + ((v * dt / (dx ** 2)) * (un[j + 1, i] - 2 * un[j, i] + un[j - 1, i])) +
                       ((v * dt / (dy ** 2)) * (un[j, i + 1] - 2 * un[j, i] + un[j, i - 1])))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
    return u


# function of frame update
def animate(frame):
    global u
    if frame > 0:
        math_parth(u, un, c, dt, dx, nx, dy, ny,v)
        im.set_array(u)
    return im,


# calling animation
ani = FuncAnimation(fig, animate, frames=nt, interval=0, blit=False)
#ani.save("2D_Diffusion.gif", writer='pillow',fps=60)
plt.show()
