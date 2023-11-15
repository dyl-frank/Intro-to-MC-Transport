import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import threading
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from tabulate import tabulate

time_0 = time.time()
seed = time.time()//10
random.seed(seed)

print(f"Starting Monte Carlo Simulation with Random Seed: {seed}")
# Specify many particles to initialize
numParticles = 10000

# Nuclear data fro Pu-238
rho_Pu239  = 0.04997                                        # Atom denity of Pu-239 (PNNL)
SigA_Pu239 = (2.403557 + 7.867439*10**(-4)) * rho_Pu239     # Macroscopic capture cross section of Pu-239 (ENDF/B-VI.8)
SigF_Pu239 = (2.403557) * rho_Pu239                         # Macroscopic fission cross section of Pu-239 (ENDF/B-VI.8)
SigS_Pu239 = (2.728796) * rho_Pu239                         # Macroscopic scattering cross section of Pu-239 (ENDF/B-VI.8)
SigT_Pu239 = SigA_Pu239 + SigS_Pu239                        # Total macroscopic capture cross section of Pu-239
lambda_Pu239 = 1/SigT_Pu239                                 # Total mean free path of Pu-239

# Nuclear data for H2O. Note that the data is compiled for 1 part oxygen, 2 parts hydrogen
rho_H2O    = 0.1
SigA_H2O   = ((2.956611*10**(-5))*0.667  + (2.834901*10**(-5) + 0.04212179 + 6.04765*10**(-14) + 0.1416575)*0.333) * rho_H2O
SigS_H2O   = ((0.6875623)*0.667  + (0.9565823)*0.333 ) * rho_H2O
SigT_H2O = SigA_H2O + SigS_H2O
lambda_H2O = 1/SigT_H2O

# Defining the cell boundaries
R1 = 3
R2 = 5
R3 = 7

nCell1 = 0
nCell2 = 0
nCell3 = 0
nFissions = 0



class Particle:

    def __init__(self, dead, x, y, z, nScatters, F):
        self.originX_set = False   
        self.originY_set = False
        self.originZ_set = False

        # Most recent position of the neutron
        self.x = x
        self.y = y
        self.z = z

        self.nScatters = nScatters       # Number of times the neutron has scattered
        self.F = F                       # Whether or not the neutron came from a fission
        self.dead = dead                 # Whether or not the neutron has been captured

        if not self.originX_set:
            self.originX = x
            Particle.originX_set = True

        if not self.originY_set:
            self.originY = y
            self.originY_set = True
        
        if not self.originZ_set:
            self.originZ = z
            self.originZ_set = True
        
    
    def r(self):                                                    
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)   # Return the neutron's distance from the origin



def Plutonium(particle):
    global nCell1, nCell3, nFissions

    s = -lambda_Pu239 * np.log(random.random())     # Sampling a exponentially distributed random step length with a mean of Pu-239's mean free path
    
    # Randomly sample longitude and lattitude
    phi = np.arccos(-1 + random.random()*2)
    theta = random.random()*2*np.pi

    r0 = particle.r()

    # Generate the stepsize in cartesian directions
    dx = s * np.sin(theta) * np.cos(phi)
    dy = s * np.sin(theta) * np.sin(phi)
    dz = s * np.cos(theta)
    
    r_new = np.sqrt((particle.x + dx)**2 + (particle.y + dy)**2 + (particle.z + dz)**2)

    # Solve for the longitude and latitude of the resultant vector
    theta_new = np.arctan((particle.y + dy ) / (particle.x + dx))
    phi_new   = np.arccos((particle.z + dz) / r_new)

    # Conditional statements checking if the resultant vector is outside the bounds of the original cell
    #   If so, translate the particle to the boundary of the neighboring cell in the direction of the resultant vector, and return to proccess_particle
    if r0 <= R1:
        if R1 <= r_new:
            particle.x = (R1 + 0.0001) * np.sin(phi_new)*np.cos(theta_new)
            particle.y = (R1 + 0.0001) * np.sin(phi_new)*np.sin(theta_new)
            particle.z = (R1 + 0.0001) * np.cos(phi_new)
            return
        
    elif R2 <= r0 <= R3:
        
        if r_new <= R2:
            particle.x = (R2 - 0.0001) * np.sin(phi_new)*np.cos(theta_new)
            particle.y = (R2 - 0.0001) * np.sin(phi_new)*np.sin(theta_new)
            particle.z = (R2 - 0.0001) * np.cos(phi_new)
            return
        
        elif R3 <= r_new:
            particle.x = (R3 + 0.0001) * np.sin(phi_new)*np.cos(theta_new)
            particle.y = (R3 + 0.0001) * np.sin(phi_new)*np.sin(theta_new)
            particle.z = (R3 + 0.0001) * np.cos(phi_new)
            return

    # If the particle remains in the cell that it started in, update the position values
    particle.x += dx
    particle.y += dy
    particle.z += dz

    # Sampling if the particle is absorbed
    if random.random() < SigA_Pu239/SigT_Pu239:
        
        # If absorbed, sample it for fission. If fission then kill the original particle and spawn 3 prompt neutrons. Otherwise just kill the particle
        if random.random() < SigF_Pu239/SigA_Pu239:
            nFissions += 1
            for i in range(3):
                prompt = Particle(dead= 0, x= particle.x, y= particle.y, z= particle.z , nScatters=0, F=1)
                particle_list.append(prompt)
        particle.dead = True
        if particle.r() <= R1:
            nCell1+=1
        else:
            nCell3+=1

    else:
        particle.nScatters+=1
        return


def Water(particle):
    global nCell2
    s = -lambda_Pu239 * np.log(random.random())     # Sampling a exponentially distributed random step length with a mean of Pu-239's mean free path
    
    # Randomly sample longitude and lattitude
    phi = np.arccos(-1 + random.random()*2)
    theta = random.random()*2*np.pi
    r0 = particle.r()

    # Generate the stepsize in cartesian directions
    dz = s*np.cos(phi)
    dx = s*np.sin(phi)*np.cos(theta)
    dy = s*np.sin(phi)*np.sin(theta)
    
    r_new = np.sqrt((particle.x + dx)**2 + (particle.y + dy)**2 + (particle.z + dz)**2)

    # Solve for the longitude and latitude of the resultant vector
    theta_new = np.arctan((particle.y + dy ) / (particle.x + dx))
    phi_new   = np.arccos((particle.z + dz) / r_new)

    if R1 <= r0 <= R2:
        if r_new <= R1:
            particle.x = (R1 - 0.0001) * np.sin(phi_new)*np.cos(theta_new)
            particle.y = (R1 - 0.0001) * np.sin(phi_new)*np.sin(theta_new)
            particle.z = (R1 - 0.0001) * np.cos(phi_new)
            return
        elif R2 <= r_new:
            particle.x = (R2 + 0.0001) * np.sin(phi_new)*np.cos(theta_new)
            particle.y = (R2 + 0.0001) * np.sin(phi_new)*np.sin(theta_new)
            particle.z = (R2 + 0.0001) * np.cos(phi_new)
            return
        
    elif R3 <= r0:
        if r_new <= R3:
            particle.x = (R3 - 0.001) * np.sin(phi_new)*np.cos(theta_new)
            particle.y = (R3 - 0.001) * np.sin(phi_new)*np.sin(theta_new)
            particle.z = (R3 - 0.001) * np.cos(phi_new)
            return
        
    particle.x += dx
    particle.y += dy
    particle.z += dz



    if random.random() < SigA_H2O/SigT_H2O:
        particle.dead = True
        if R1 <= particle.r() <= R2:
            nCell2+=1
    else:
        particle.nScatters+=1
        return

def process_particle(particle):
    while not particle.dead:
        if particle.r() < R1:
            Plutonium(particle=particle)
        elif R1 < particle.r() < R2:
            Water(particle=particle)
        elif R2 < particle.r() < R3:
            Water(particle=particle)
        elif particle.r() > R3:
            Water(particle=particle)

particle_list = []
# Initialize the particles
for i in range(numParticles):
    particle = Particle(dead= 0, x= 0, y= 0, z= 0, nScatters=0, F=0)
    particle_list.append(particle)


# Define the number of threads (adjust this based on your system)
num_threads = 6

# Initialize a list to hold the threads
threads = []

for i, particle in enumerate(particle_list):
    # Create a thread for each particle
    thread = threading.Thread(target=process_particle, args=(particle,))
    threads.append(thread)
    thread.start()
    if time.time()- time_0 > 1000 or len(particle_list) > 1000000:
        print("Time out!")
        break

# Wait for all threads to finish
for thread in threads:
    thread.join()

print(f"Simulation Complete!\nCompute Time:     {time.time()-time_0} s\nTotal Number of Particles Simulated: {len(particle_list)}\nTotal Number of Fissions: {nFissions}\nFissions per Initialized Particle: {nFissions/numParticles}")

table_data = [
    ['Cell', 'Number of Captures', 'Density (neutrons per cm^3)', 'Flux (Neutrons per cm^2 per second)'],
    ['Cell 1', nCell1, nCell1/((4/3)*np.pi*R1**3), nCell1/((4/3)*np.pi*R1**3)*5.2*10**9],
    ['Cell 2', nCell2, nCell2/((4/3)*np.pi*R1**3), nCell2/((4/3)*np.pi*(R2**3-R1**3))*5.2*10**9],
    ['Cell 3', nCell3, nCell3/((4/3)*np.pi*R1**3), nCell3/((4/3)*np.pi*(R3**3-R2**3))*5.2*10**9]
]
print(tabulate(table_data, headers='firstrow', tablefmt='fancy_grid'))





# #####################################################################
#                       PLOTING
# #####################################################################


def plot_capture_3D(particle_list):

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('grey')
    ax.view_init(elev=27, azim=-75) 
    # Get the maximum distance from the origin
    max_distance = max(np.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2) for p in particle_list)

    # Loop through particles and plot them one by one
    for particle in random.sample(particle_list, len(particle_list)//100):
        distance = np.sqrt(particle.x ** 2 + particle.y ** 2 + particle.z ** 2)
        normalized_distance = distance / max_distance
        color = plt.cm.plasma(normalized_distance)  # Use plasma colormap

        ax.scatter(particle.x, particle.y, particle.z, c=color, marker='o', s=0.1)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a colorbar with logarithmic scale
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, max_distance))
    sm.set_array([])
    plt.colorbar(sm, label='Distance from Origin (log scale)')

    # Set axis labels and ticks color to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')

    # Set the color of the ticks
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Add a title
    plt.title("Positions of Captured Neutrons (cm from origin)", color='black')

    # Show the plot
    plt.show()

def plot_origin_3D(particle_list):
     # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('grey')
    ax.view_init(elev=27, azim=-75) 
    # Get the maximum distance from the origin
    max_distance = max(np.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2) for p in particle_list)

    # Loop through particles and plot them one by one
    for particle in random.sample(particle_list, len(particle_list)//100):
        distance = np.sqrt(particle.originX ** 2 + particle.originY ** 2 + particle.originZ ** 2)
        normalized_distance = distance / R3
        color = plt.cm.plasma(normalized_distance)  # Use plasma colormap

        ax.scatter(particle.originX, particle.originY, particle.originZ, c=color, marker='o', s=0.1)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a colorbar with logarithmic scale
    sm = plt.cm.ScalarMappable(cmap='plasma')
    sm.set_array([])
    norm = plt.Normalize(0, 9)
    plt.colorbar(sm, label='Distance from Origin (cm)', norm=norm)

    # Set axis labels and ticks color to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')

    # Set the color of the ticks
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Add a title
    plt.title("Spawn Point for Neutrons (cm from origin)", color='black')

    # Show the plot
    plt.show()

def plot_tracks_3D(particle_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('grey')

    # Get maximum scatter
    max_nScatters = max(p.nScatters for p in particle_list)
    # Set axis limits
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])

    # Randomly select 100 particles
    selected_particles = random.sample(particle_list, len(particle_list))
    # Loop through selected particles and plot arrows
    for particle in selected_particles:
        if particle.F:
            ax.quiver(particle.originX, particle.originY, particle.originZ,
                    particle.x - particle.originX, particle.y - particle.originY, particle.z - particle.originZ,
                    color=plt.cm.plasma(particle.nScatters/30), linewidth=0.5, arrow_length_ratio=0.1)
        if particle.F:
            ax.scatter(particle.originX, particle.originY, particle.originZ, c="red", marker='o', s=0.6)


    # Adding colorbar
    norm = plt.Normalize(0, 30)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Number of Times Scattered')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    # Set axis labels and ticks color to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')

    # Set the color of the ticks
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    plt.title("Resultant Translation of Neutrons (cm from origin)", color='black')
    # Show the plot
    plt.show()



def plot_capture(particle_list):
    # Extract data for plotting
    x = [particle.x for particle in particle_list]
    y = [particle.y for particle in particle_list]
    r = [ np.sqrt(particle.x**2 + particle.y**2) for particle in particle_list]

    # Create a scatter plot with color coding by nScatters
    plt.scatter(x, y, c=r, cmap='plasma', s=0.1)

    # Adding colorbar
    plt.colorbar(label='Distance from Origin (cm)')

    # Set labels for the axes
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title("Captured Neutrons Projected onto XY (cm from origin)", color='black')

    # Show the plot
    plt.show()


def plot_origin(particle_list):
    # Extract data for plotting
    x = [particle.originX for particle in particle_list]
    y = [particle.originY for particle in particle_list]
    r = [np.sqrt(particle.originX**2 + particle.originY**2) for particle in particle_list]

    # Create a scatter plot with color coding by nScatters
    plt.scatter(x, y, c=r, cmap='plasma', s=0.1)

    # Adding colorbar
    plt.colorbar(label='Distance from Origin (cm)')

    # Set labels for the axes
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title("Spawn Point for Neutrons Projected onto XY (cm from origin)", color='black')

    # Show the plot
    plt.show()


def plot_tracks(particle_list):
    # Loop through particles and plot arrows
    for particle in random.sample(particle_list, len(particle_list)//10):
        plt.arrow(particle.originX, particle.originY, particle.x - particle.originX, particle.y - particle.originY,
                  color=plt.cm.plasma(particle.nScatters/30), head_width=0.6, length_includes_head=True, linewidth=0.4)

    # Adding colorbar
    norm = plt.Normalize(0, 30)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Number of Times Scattered')
    # Set labels for the axes
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Resultant Translation of Neutrons, Projected onto XY (cm from origin)", color='black')
    # Show the plot
    plt.show()

plot_capture(particle_list=particle_list)
plot_origin(particle_list=particle_list)
plot_tracks(particle_list=particle_list)
plot_capture_3D(particle_list=particle_list)
plot_origin_3D(particle_list=particle_list)
plot_tracks_3D(particle_list=particle_list)