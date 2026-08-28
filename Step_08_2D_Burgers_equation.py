import os
import numpy as np
from pyevtk.hl import gridToVTK
import shutil

# Folder to save results ParaView
output_dir = "paraview_Burgers_results"

# Delete old data
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a new folder
os.makedirs(output_dir, exist_ok=True)

# Variable declarations
nx = 81
ny = 81
nz = 1

dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
nu = 0.01
sigma = .009
dt = sigma * dx * dy / nu

nt = 1201
save_every = 10

# Arrays initialization
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
z = np.zeros(nz)

u = np.ones((nx, ny))
v = np.ones((nx, ny))

# Set hat function
u[int(.5 / dx):int(1. / dx + 1), int(.5 / dy):int(1. / dy + 1)] = 2
v[int(.5 / dx):int(1. / dx + 1), int(.5 / dy):int(1. / dy + 1)] = 2

print("Simulation started, computing solution... ")


# Main Burgers solver
def math_part(u, v, dt, dx, dy, nu, x, y, z, output_dir):
    w = np.zeros_like(u)
    for it in range(nt):
        un = u.copy()
        vn = v.copy()

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         (un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[:-2, 1:-1])) -
                         (vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[1:-1, :-2])) +
                         ((nu * dt / (dx ** 2)) * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])) +
                         ((nu * dt / (dy ** 2)) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         (un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])) -
                         (vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[1:-1, :-2])) +
                         ((nu * dt / (dx ** 2)) * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])) +
                         ((nu * dt / (dy ** 2)) * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2])))

        # Boundary conditions
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

        # Save data at specified intervals for ParaView
        if it % save_every == 0:
            file_idx = it // save_every
            filepath = os.path.join(output_dir, f"Burgers_{file_idx:04d}")

            gridToVTK(
                filepath, x, y, z,
                pointData={
                    "Velocity": (u[..., np.newaxis], v[..., np.newaxis], w)
                }
            )
    return u, v


u, v = math_part(u, v, dt, dx, dy, nu, x, y, z, output_dir)
print(f"Ready, files updated in folder: {output_dir}")
