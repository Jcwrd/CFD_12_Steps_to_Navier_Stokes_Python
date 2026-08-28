import os
import numpy as np
from pyevtk.hl import gridToVTK
import shutil

# Folder to save results ParaView
output_dir = "paraview_nonlinear_convection_results"

# Delete old data
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a new folder
os.makedirs(output_dir, exist_ok=True)

# Variable declarations
nx = 201
ny = 201
nz = 1

dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

nt = 501
save_every = 10

# Arrays initialization
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
z = np.zeros(nz)

u = np.ones((nx, ny))
un = np.ones((nx, ny))

# set hat function
u[int(.25 / dx):int(.75 / dx + 0.5), int(.25 / dy):int(.75 / dy + 0.5)] = 2

print("Simulation started, computing solution... ")


# Main Unlinear convection solver
def math_part(u, un, dt, dx, dy, x, y, z, output_dir):
    vy = np.zeros_like(u)
    w = np.zeros_like(u)
    for it in range(nt):
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - (un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[:-2, 1:-1])) -
                         (un[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[1:-1, :-2])))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        # Save data at specified intervals for ParaView
        if it % save_every == 0:
            file_idx = it // save_every
            filepath = os.path.join(output_dir, f"nonlinear_convection_{file_idx:04d}")

            gridToVTK(
                filepath, x, y, z,
                pointData={
                    "Velocity": (u[..., np.newaxis], vy, w)
                }
            )
    return u


u = math_part(u, un, dt, dx, dy, x, y, z, output_dir)
print(f"Ready, files updated in folder: {output_dir}")
