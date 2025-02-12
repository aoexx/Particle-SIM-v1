import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# Simulation parameters
dt = 0.005  # Increased time step for better motion
num_steps = 500
num_particles = 20
box_size = 10
epsilon = 1.0
sigma = 1.0
mass = 1.0
cutoff_radius = 2.5 * sigma

# Initialize particle positions and velocities
np.random.seed(42)
positions = np.random.uniform(2, box_size - 2, (num_particles, 3))  # Ensure particles are spread out
velocities = np.random.uniform(-0.5, 0.5, (num_particles, 3))
velocities[np.linalg.norm(velocities, axis=1) < 0.1] += 0.1  # Ensure a minimum velocity
forces = np.zeros_like(positions)

# Lennard-Jones force function
def lennard_jones_force(r):
    r_mag = np.linalg.norm(r)
    if r_mag == 0 or r_mag > cutoff_radius:
        return np.zeros(3)
    force_magnitude = 24 * epsilon * ((2 * (sigma ** 12) / (r_mag ** 13)) - ((sigma ** 6) / (r_mag ** 7)))
    return force_magnitude * (r / r_mag)

# Compute forces between particles
def compute_forces(positions):
    forces = np.zeros_like(positions)  # Ensure forces is a NumPy array
    potential_energy = 0
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r_ij = positions[j] - positions[i]
            r_mag = np.linalg.norm(r_ij)
            if r_mag < cutoff_radius:
                force = lennard_jones_force(r_ij)
                forces[i] += force
                forces[j] -= force
                potential_energy += 4 * epsilon * ((sigma / r_mag) ** 12 - (sigma / r_mag) ** 6)  # Energy calculation
    return forces, potential_energy  # Ensure two values are returned

# Initialize arrays for storing simulation data
positions_list = [positions.copy()]
unwrapped_positions = positions.copy()  # Track unwrapped positions
velocities_list = [velocities.copy()]

# Initial force calculation
forces, potential_energy = compute_forces(positions)

# Run simulation
for step in range(num_steps):
    positions = np.array(positions)  # Ensure positions is a NumPy array
    velocities = np.array(velocities)  # Ensure velocities is a NumPy array
    forces = np.array(forces)  # Ensure forces is a NumPy array
    
    positions += velocities * dt + 0.5 * forces * (dt ** 2) / mass
    new_forces, potential_energy = compute_forces(positions)
    new_forces = np.array(new_forces)  # Ensure new_forces is a NumPy array
    
    velocities += 0.5 * (forces + new_forces) * dt / mass
    forces = new_forces
    unwrapped_positions += velocities * dt  # Track unwrapped positions
    positions = positions % box_size  # Apply periodic boundary conditions
    positions_list.append(unwrapped_positions.copy())
    velocities_list.append(velocities.copy())

positions_array = np.array(positions_list)

# Reduce figure size and switch to 2D visualization to optimize rendering
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
colors = sns.color_palette("husl", num_particles)
scatters = [ax.scatter([], [], color=colors[i]) for i in range(num_particles)]

ax.set_xlim([0, box_size])
ax.set_ylim([0, box_size])
ax.set_xlabel("X", fontsize=12, fontweight='bold')
ax.set_ylabel("Y", fontsize=12, fontweight='bold')
ax.set_title("Particle Simulation (2D Projection)", fontsize=14, fontweight='bold')

# Update function for animation (2D projection in X-Y plane)
def update_2d(frame):
    for i, scatter in enumerate(scatters):
        scatter.set_offsets([positions_array[frame, i, 0], positions_array[frame, i, 1]])
    return scatters

# Create animation with optimized frame rate
ani_2d = animation.FuncAnimation(fig, update_2d, frames=num_steps, interval=30, blit=False)

# Save animation as a video file
video_path = "particle_simulation.gif"
ani_2d.save(video_path, writer="pillow", fps=20)

print("Animation saved as", video_path)