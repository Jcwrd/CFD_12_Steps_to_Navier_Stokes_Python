import os
import numpy as np
from pyevtk.hl import gridToVTK
import shutil

# Folder to save results ParaView
output_dir = "paraview_Diffusion_1D_results"

# Delete old data
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a new folder
os.makedirs(output_dir, exist_ok=True)

# Variable declarations
nx = 101
ny = 1
nz = 1
dx = 2 / (nx - 1)
nu = 0.3
sigma = .2
dt = sigma * dx ** 2 / nu
nt = 4000
save_every = 20

# Arrays initialization
x = np.linspace(0, 2, nx)
y = np.zeros(ny)
z = np.zeros(nz)

# Initialize a pure 1D array
u = np.zeros(nx)
un = np.zeros(nx)

u[int(.75 / dx):int(1 / dx + 1)] = 2

print("Simulation started, computing solution... ")


# Main Diffusion equation solver
def math_part(u, un, dt, nt, dx, nu, save_every, x, y, z, output_dir):
    for it in range(nt):
        un = u.copy()

        u[1:-1] = un[1:-1] + nu * dt / dx ** 2 * (un[2:] - 2 * un[1:-1] + un[:-2])

        # Save data at specified intervals for ParaView
        if it % save_every == 0:
            file_idx = it // save_every
            filepath = os.path.join(output_dir, f"Diffusion_1D_{file_idx:04d}")

            gridToVTK(
                filepath, x, y, z,
                pointData={
                    # PyEVTK requires shape like (nx, ny, nz), add fictitious axes.
                    "Velocity": u[:, np.newaxis, np.newaxis]
                }
            )
    return u


math_part(u, un, dt, nt, dx, nu, save_every, x, y, z, output_dir)
print(f"Ready, files updated in folder: {output_dir}")

# --- ParaView Visualization Instructions ---
# 1. Open the data file series in the 'paraview_Diffusion_1D_results' directory.
# 2. Click 'Apply' in the Properties panel.
# 3. Go to Filters -> Data Analysis -> Plot Over Line (or Filters -> Plot Over Line).
# 4. Configure the line coordinates and apply to see the 2D XY plot.
