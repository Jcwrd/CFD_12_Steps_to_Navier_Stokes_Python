import numpy
import numba
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# variable declarations
nx = 61
ny = 61
nt = 121
c = 2
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
nu = 0.01
sigma = .009
dt = sigma * dx * dy / nu

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

u = numpy.ones((ny, nx))
v = numpy.ones((ny, nx))
un = numpy.ones((ny, nx))
vn = numpy.ones((ny, nx))
ultimate = numpy.ones((ny, nx))
# Assign initial conditions

# set hat function
u[int(.5 / dy):int(1. / dy + 1), int(.5 / dx):int(1. / dx + 1)] = 2 * numpy.sqrt(2)
v[int(.5 / dy):int(1. / dy + 1), int(.5 / dx):int(1. / dx + 1)] = 2 * numpy.sqrt(2)
# figure setting
fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(u, cmap='viridis', extent=[0, 2, 0, 2], origin='lower', vmin=numpy.sqrt(2), vmax=2 * numpy.sqrt(2))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('u')

ax.set_title('2D Burger`s equation Field')
ax.set_xlabel('X')
ax.set_ylabel('Y')


@numba.njit
def math_parth(u, v, vn, un, c, dt, dx, nx, dy, ny, nu):
    un = u.copy()
    vn = v.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] = (un[j, i] -
                       (un[j, i] * dt / dx * (un[j, i] - un[j, i - 1])) -  # шаг по X (i-1)
                       (vn[j, i] * dt / dy * (un[j, i] - un[j - 1, i])) +  # шаг по Y (j-1)
                       ((nu * dt / (dx ** 2)) * (un[j, i + 1] - 2 * un[j, i] + un[j, i - 1])) +
                       ((nu * dt / (dy ** 2)) * (un[j + 1, i] - 2 * un[j, i] + un[j - 1, i])))

            v[j, i] = (vn[j, i] -
                       (un[j, i] * dt / dx * (vn[j, i] - vn[j, i - 1])) -  # шаг по X (i-1)
                       (vn[j, i] * dt / dy * (vn[j, i] - vn[j - 1, i])) +  # шаг по Y (j-1)
                       ((nu * dt / (dx ** 2)) * (vn[j, i + 1] - 2 * vn[j, i] + vn[j, i - 1])) +
                       ((nu * dt / (dy ** 2)) * (vn[j + 1, i] - 2 * vn[j, i] + vn[j - 1, i])))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1
    return u, v


# function of frame update
def animate(frame):
    global u
    global v
    if frame > 0:
        math_parth(u, v, vn, un, c, dt, dx, nx, dy, ny, nu)
        ultimate = numpy.sqrt(v ** 2 + u ** 2)
        im.set_array(ultimate)
    return im,


# calling animation
ani = FuncAnimation(fig, animate, frames=nt, interval=0, blit=True)
# ani.save("2D_Burger`s_equation.gif", writer='pillow',fps=60)
plt.show()
