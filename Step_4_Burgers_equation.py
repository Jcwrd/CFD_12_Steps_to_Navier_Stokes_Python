import os
import sympy
import numpy as np
from pyevtk.hl import gridToVTK
import shutil
from sympy import lambdify

# Folder to save results ParaView
output_dir = "paraview_Burgers_1D_results"

# Delete old data
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a new folder
os.makedirs(output_dir, exist_ok=True)

# Variable declarations
nx = 201
ny = 1
nz = 1
dx = 2 * np.pi / (nx - 1)
nu_val = 0.01
dt = dx * nu_val
nt = 20000
save_every = 100

# Variable roles must be split into SymPy symbols _sym and pure values _val
x_sym, nu_sym, t_sym = sympy.symbols('x_sym nu_sym t_sym')

phi = (sympy.exp(-(x_sym - 4 * t_sym) ** 2 / (4 * nu_sym * (t_sym + 1))) +
       sympy.exp(-(x_sym - 4 * t_sym - 2 * sympy.pi) ** 2 / (4 * nu_sym * (t_sym + 1))))
phiprime = phi.diff(x_sym)

u_expr = -2 * nu_sym * (phiprime / phi) + 4
ufunc = lambdify((t_sym, x_sym, nu_sym), u_expr)

# Arrays initialization
x = np.linspace(0, 2 * np.pi, nx)
y = np.zeros(ny)
z = np.zeros(nz)

# Initialize a pure 1D array
t_init = 0
u = ufunc(t_init, x, nu_val)
un = np.zeros(nx)
u[-1] = u[0]

print("Simulation started, computing solution... ")


# Main Burgers equation solver
def math_part(u, un, dt, dx, x, y, z, output_dir):
    for it in range(nt):
        un = u.copy()

        u[1:-1] = (un[1:-1] -
                   un[1:-1] * dt / dx * (un[1:-1] - un[:-2]) +
                   nu_val * dt / dx ** 2 * (un[2:] - 2 * un[1:-1] + un[:-2]))
        # Periodic boundary conditions
        u[0] = (un[0] -
                un[0] * dt / dx * (un[0] - un[-2]) +
                nu_val * dt / dx ** 2 * (un[1] - 2 * un[0] + un[-2]))
        u[-1] = u[0]

        # Save data at specified intervals for ParaView
        if it % save_every == 0:
            file_idx = it // save_every
            filepath = os.path.join(output_dir, f"Burgers_1D_{file_idx:04d}")

            # PyEVTK requires shape like (nx, ny, nz), add fictitious axes.

            gridToVTK(
                filepath, x, y, z,
                pointData={
                    # PyEVTK requires shape like (nx, ny, nz), add fictitious axes.
                    "Velocity": u[:, np.newaxis, np.newaxis]
                }
            )
    return u


math_part(u, un, dt, dx, x, y, z, output_dir)
print(f"Ready, files updated in folder: {output_dir}")

# --- ParaView Visualization Instructions ---
# 1. Open the data file series in the 'paraview_Burgers_1D_results' directory.
# 2. Click 'Apply' in the Properties panel.
# 3. Go to Filters -> Data Analysis -> Plot Over Line (or Filters -> Plot Over Line).
# 4. Configure the line coordinates and apply to see the 2D XY plot.
