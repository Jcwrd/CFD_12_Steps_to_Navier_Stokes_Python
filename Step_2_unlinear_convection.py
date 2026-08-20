import numpy                 # we're importing numpy
from matplotlib import pyplot, pyplot as plt  # and our 2D plotting library
from matplotlib.animation import FuncAnimation


nx = 101
dx = 7 / (nx - 1)
nt = 50    #nt is the number of timesteps we want to calculate
dt = 0.005  #dt is the amount of time each timestep covers (delta t)

u = numpy.ones(nx)      #as before, we initialize u with every value equal to 1.
u[int(.5 / dx) : int(1 / dx + 1)] = 2  #then set u = 2 between 0.5 and 1 as per our I.C.s

fig , (ax1,ax2) = plt.subplots(2,1,figsize=(12,6),gridspec_kw={'height_ratios':[2,1]})  #that create window with 2 graphics

#setting first graphic of velocity
x_coords=numpy.linspace(0,5,nx) #creating uniform array with nx-th points of x coords our grid
line,= ax1.plot(x_coords,u,lw=2,color="darkblue") #building graphic with x - x_coords y - u, thickness - 2
# we use , because it helps us with future animations( save line )
ax1.set_ylabel('u - Speed of vawe м/с')
ax1.set_title('Unlinear Convection')
ax1.grid(True,linestyle='--',alpha=0.5) #turn on the grid with set type of line and transparency

#setting second graphic
pipe_data=numpy.atleast_2d(u) #remake linear array to 2D array
im=ax2.imshow(pipe_data,cmap='viridis',extent=[0,5,0,0.1],vmin=1,vmax=2) #show our array 0-5 on horizontally 0-0.1 on vertically
ax2.set_xlim(0,5)
ax2.set_ylim(0,0.1)
ax2.set_xlabel('coordinate x on pipe')

#add colorbar
cbar = fig.colorbar(im,ax=[ax1,ax2],orientation='vertical',pad=0.05,shrink=1)
cbar.set_label('скорость течения')
#un = numpy.ones(nx) #initialize our placeholder array un, to hold the time-stepped solution
def animate(frame):
    global u
    if frame == 0:
        u = numpy.ones(nx)
        u[int(0.5 / dx): int(1.0 / dx + 1)] = 2
    else:
        un = u.copy()
        for i in range(1, nx):
            u[i] = un[i] - un[i] * (dt / dx) * (un[i] - un[i - 1])
    line.set_ydata(u)  # линия графика
    im.set_array(numpy.atleast_2d(u))  # цвета в трубе

    return line, im




# start interaction animation
# interval=40 — time betwine frames
ani = FuncAnimation(fig, animate, frames=1000, interval=0, blit=True, repeat=True)
ani.save('Unlinear_convection.gif',writer='pillow',fps=60)
plt.tight_layout()
pyplot.show()