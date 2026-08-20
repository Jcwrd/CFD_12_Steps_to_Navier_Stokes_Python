import numpy
from matplotlib import pyplot, pyplot as plt
from matplotlib.animation import FuncAnimation

#from Step_2_unlinear_convection import pipe_data

domain_of_distribution=5-0
nx=101  #exact of grid
dx=domain_of_distribution/(nx-1) #time step size
nt=101 #number of time steps
dt=0.012 #length in time between initial conditions and last step
c=1 #wave velocity

u=numpy.ones(nx) #every value of array equals 1
u[int(.5 / dx):int(1 / dx + 1)] = 2 # values between 0.5 and 1 equal 2

#pyplot.plot(numpy.linspace(0,2,nx), u) # make a grid with distribution numpy... on x and u on u
# linspace generate value between 0 and 2 with 41 points

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,6),gridspec_kw={'height_ratios':[2,1]})

x_coords=numpy.linspace(0,5,nx)
line,= ax1.plot(x_coords,u,lw=2,color="darkblue")
ax1.set_xlabel('x')
ax1.set_ylabel('u')
ax1.set_title('linear Convection')
ax1.grid(True,linestyle='--',alpha=0.5)

pipe_data=numpy.atleast_2d(u)
im=ax2.imshow(pipe_data,cmap='viridis',extent=[0,5,0,0.2],vmin=1,vmax=2)

cbar=fig.colorbar(im,ax=[ax1,ax2],orientation='vertical',pad=0.05,shrink=1)
cbar.set_label('speed of convection')

def update(frame):
    global u
    if frame==0:
        u = numpy.ones(nx)  # every value of array equals 1
        u[int(.5 / dx):int(1 / dx + 1)] = 2  # values between 0.5 and 1 equal 2
    else:

        un=numpy.ones(nx) #create temporary array
        # #time loop to advance solution
        un=u.copy() #fill in helper array
        for i in range(1,nx): # inner loop to solve that equation in space
            u[i]=un[i]-(dt/dx)*c*(un[i]-un[i-1]) #no comments

    line.set_ydata(u)
    im.set_array(numpy.atleast_2d(u))
    return line, im

#pyplot.plot(numpy.linspace(0,2,nx), u) #plotting our solution
ani=FuncAnimation(fig, update, frames=400,interval=0,blit=True,repeat=True)
pyplot.tight_layout()

ani.save('linear_convection.gif',writer='pillow',fps=60)

pyplot.show() #function which illustrate graphic of distribution
