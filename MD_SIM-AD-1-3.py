# NOTE: This version uses basic PBCs instead of reflective boundaries
# See the final version (Particle-SIM-v2: MD_SIM-AD-2-1.py) for improved boundary handling

from manim import *
import numpy as np

# Simulation parameters
dt = 0.005  # Time step
num_steps = 500  # Number of time steps
num_particles = 10  # Number of particles for clarity
box_size = 10  # Box size
epsilon = 1.0  # Lennard-Jones potential parameter
sigma = 1.0  # Lennard-Jones potential parameter
mass = 1.0  # Particle mass
cutoff_radius = 2.5 * sigma  # Interaction cutoff distance

# Initialize particle positions and velocities
np.random.seed(42)
positions = np.random.uniform(2, box_size - 2, (num_particles, 3))
velocities = np.random.uniform(-0.5, 0.5, (num_particles, 3))
forces = np.zeros_like(positions)
trajectories = np.zeros((num_steps, num_particles, 3))  # Store trajectories

# Lennard-Jones force function
def lennard_jones_force(r):
    r_mag = np.linalg.norm(r)
    if r_mag == 0 or r_mag > cutoff_radius:
        return np.zeros(3)
    force_magnitude = 24 * epsilon * ((2 * (sigma ** 12) / (r_mag ** 13)) - ((sigma ** 6) / (r_mag ** 7)))
    return force_magnitude * (r / r_mag)

# Compute forces between particles
def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r_ij = positions[j] - positions[i]
            r_mag = np.linalg.norm(r_ij)
            if r_mag < cutoff_radius:
                force = lennard_jones_force(r_ij)
                forces[i] += force
                forces[j] -= force
    return forces

# Run simulation
forces = compute_forces(positions)
for step in range(num_steps):
    positions += velocities * dt + 0.5 * forces * (dt ** 2) / mass
    new_forces = compute_forces(positions)
    velocities += 0.5 * (forces + new_forces) * dt / mass
    forces = new_forces
    positions = positions % box_size  # Apply periodic boundary conditions
    trajectories[step] = positions.copy()

class ParticleSimulation(ThreeDScene):
    CONFIG = {"frame_rate": 24}  # Set FPS to 24
    
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        axes = ThreeDAxes()
        self.add(axes)

        # Create particles
        particles = [
            Sphere(radius=0.2, color=RED).move_to(trajectories[0, i])
            for i in range(num_particles)
        ]
        for p in particles:
            self.add(p)

        # Animate trajectories
        for step in range(1, num_steps, 5):  # Skip frames for speed
            self.play(*[
                p.animate.move_to(trajectories[step, i])
                for i, p in enumerate(particles)
            ], run_time=0.2)  # Increase run_time to 0.2 seconds
        
        self.wait(1)

if __name__ == "__main__":
    from manim import config
    config.background_color = BLACK  # Optional: Set background color
    scene = ParticleSimulation()
    scene.render()
