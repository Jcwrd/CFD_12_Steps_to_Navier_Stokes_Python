import os
import numpy as np
from pyevtk.hl import gridToVTK
import shutil

# Folder to save results ParaView
output_dir = "paraview_cavity_results"

# Delete old data
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a new folder
os.makedirs(output_dir, exist_ok=True)

# Variables declaration
nx = 101
ny = 101
nit = 100
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

nt = 1501
save_every = 20

# Create a grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
# 3-rd axis for ParaView
z = np.zeros(1)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Physical parameters
rho = 1
nu = 0.1
dt = 0.00005

# Arrays initialization
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))
b = np.zeros((nx, ny))




def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                            ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx) +
                             (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dy)) -
                            ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx)) ** 2 -
                            2 * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dy) *
                                 (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dx)) -
                            ((v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dy)) ** 2))
    return b


def pressure(p, dx, dy, b):
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dy ** 2 +
                          (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dx ** 2) /
                         (2 * (dx ** 2 + dy ** 2)) -
                         dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                         b[1:-1, 1:-1])

        p[-1, :] = p[-2, :]  # dp/dx = 0 by x = 2
        p[:, 0] = p[:, 1]  # dp/dy = 0 by y = 0
        p[0, :] = p[1, :]  # dp/dx = 0 by x = 0
        p[:, -1] = 0  # p = 0 by y = 2

    return p


print("Simulation started, computing solution... ")


# Main Navier-Stokes solver
def math_part(nt, u, v, dt, dx, dy, p, rho, nu, b, x, y, z, output_dir):
    # Save VTK files for ParaView (requires 3 velocity components: U, V, W)
    w = np.zeros_like(u)
    for it in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure(p, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         dt / (2 * rho * dx) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) +
                               dt / dy ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         dt / (2 * rho * dy) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) +
                               dt / dy ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])))

        # Boundary conditions
        u[:, 0] = 0
        u[0, :] = 0
        u[-1, :] = 0
        u[:, -1] = 1  # Lid velocity
        v[:, 0] = 0
        v[:, -1] = 0
        v[0, :] = 0
        v[-1, :] = 0

        # Save data at specified intervals for ParaView
        if it % save_every == 0:
            file_idx = it // save_every
            filepath = os.path.join(output_dir, f"cavity_{file_idx:04d}")

            gridToVTK(
                filepath, x, y, z,
                pointData={
                    "Pressure": p[..., np.newaxis],
                    "Velocity": (u[..., np.newaxis], v[..., np.newaxis], w[..., np.newaxis]),
                }
            )

    return u, v, w


math_part(nt, u, v, dt, dx, dy, p, rho, nu, b, x, y, z, output_dir)
print(f"Ready, files updated in folder: {output_dir}")
