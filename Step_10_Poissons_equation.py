import os
import numpy as np
from pyevtk.hl import gridToVTK
import shutil

# Folder to save results ParaView
output_dir = "paraview_Poisson_results"

# Delete old data
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a new folder
os.makedirs(output_dir)

# Variable declarations
nx = 51
ny = 51
nz = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

nt = 401
save_every = 2

# Arrays initialization
p = np.zeros((nx, ny))
b = np.zeros((nx, ny))
pn = np.zeros((nx, ny))

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
z = np.zeros(nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Initial conditions
b[int(ny / 4):int(ny / 3), int(nx / 4):int(nx / 3)] = 100
b[int(3 * ny / 4):int(3 * ny / 3), int(3 * nx / 5):int(3 * nx / 4)] = -100

print("Simulation started, computing solution... ")


# Main Poisson solver
def math_part(p, pn, dx, dy, b, x, y, output_dir):
    for it in range(nt):
        pn = p.copy()

        p[1:-1, 1:-1] = (((pn[2:, 1:-1] + pn[:-2, 1:-1]) * dy ** 2 +
                          (pn[1:-1, 2:] + pn[1:-1, :-2]) * dx ** 2 -
                          b[1:-1, 1:-1] * dx ** 2 * dy ** 2) /
                         (2 * (dx ** 2 + dy ** 2)))
        # Boundary conditions
        # Neumann conditions: dp/dy = 0 @ y = 0 and y = 1
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        # Dirichlet conditions: p = 0 @ x = 0 and x = 1
        p[0, :] = 0
        p[-1, :] = 0

        # Save data at specified intervals for ParaView
        if it % save_every == 0:
            file_idx = it // save_every
            filepath = os.path.join(output_dir, f"Poisson_{file_idx:04d}")
            gridToVTK(
                filepath, X, Y, Z,
                pointData={
                    "Pressure": p[..., np.newaxis]
                }
            )
    return p


p = math_part(p, pn, dx, dy, b, x, y, output_dir)
print(f"Ready, files updated in folder: {output_dir}")
