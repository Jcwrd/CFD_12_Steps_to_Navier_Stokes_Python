import numpy
# import numba
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# variable declarations
nx = 31
ny = 31
nt = 121
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

p = numpy.zeros((ny, nx))

pn = numpy.zeros((ny, nx))

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

# figure setting
fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(p, cmap='viridis', extent=[0, 2, 0, 2], origin='lower', vmin=0, vmax=2)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('u')

ax.set_title('2D_Laplas equation')
ax.set_xlabel('X')
ax.set_ylabel('Y')


# @numba.njit
def math_parth(p, pn, dx, nx, dy, ny):
    pn = p.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            p[j, i] = (dy ** 2 * (pn[j + 1, i] + pn[j - 1, i]) + dx ** 2 * (pn[j, i + 1] + pn[j, i - 1])) / (
                    2 * (dx ** 2 + dy ** 2))
    p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
    p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
    p[:, 0] = 0
    p[:, -1] = y

    return p


# function of frame update
def animate(frame):
    global p
    if frame > 0:
        math_parth(p, pn, dx, nx, dy, ny)
        im.set_array(p)
    return im,


# calling animation
ani = FuncAnimation(fig, animate, frames=nt, interval=0, blit=True)
#ani.save("2D_Laplas_equation.gif", writer='pillow',fps=60)
plt.show()
