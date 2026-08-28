import os
import numpy as np
from pyevtk.hl import gridToVTK
import shutil

# Folder to save results ParaView
output_dir = "paraview_diffusion_results"

# Delete old data
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a new folder
os.makedirs(output_dir, exist_ok=True)

# Variable declarations
nx = 71
ny = 71
nz = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
v = 0.05

nt = 351
save_every = 1

sigma = .25
dt = sigma * dx * dy / v

# Arrays initialization
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
z = np.zeros(nz)

u = np.ones((ny, nx))
un = np.ones((ny, nx))

# Set hat function
u[int(.75 / dy):int(1.25 / dy + 1), int(.5 / dx):int(1.25 / dx + 1)] = 2

print("Simulation started, computing solution... ")


# Main diffusion function
def math_part(u, dt, dx, dy, v, x, y, z, output_dir):
    vy = np.zeros_like(u)
    w = np.zeros_like(u)
    for it in range(nt):
        un = u.copy()

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                         ((v * dt / (dx ** 2)) * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])) +
                         ((v * dt / (dy ** 2)) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        # Save data at specified intervals for ParaView
        if it % save_every == 0:
            file_idx = it // save_every
            filepath = os.path.join(output_dir, f"diffusion_{file_idx:04d}")

            gridToVTK(
                filepath, x, y, z,
                pointData={
                    "Velocity": (u[..., np.newaxis], vy, w)
                }
            )
    return u


u = math_part(u, dt, dx, dy, v, x, y, z, output_dir)
print(f"Ready, files updated in folder: {output_dir}")
