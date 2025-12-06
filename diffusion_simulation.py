import numpy as np
import matplotlib.pyplot as plt

class DiffusionSimulation:
    def __init__(self, n_particles=1000, n_steps=100, step_size=1.0):
        """
        Initialize the simulation parameters.
        """
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.step_size = step_size
        
        # These will hold the simulation data after run() is called
        
        # Rows represent the Particles. axis 0: Particles
        # Columns represent the Time Steps. axis 1: Time Steps
        self.x_pos = None
        self.y_pos = None
        self.time_axis = np.arange(n_steps + 1)

    def run_random_walk(self):
        """
        Generates 2D random walks for all particles simultaneously using vectorization.
        """
        # 1. Generate random steps (dx, dy)
        # Shape: (n_particles, n_steps)
        dx = np.random.normal(0, self.step_size, size=(self.n_particles, self.n_steps))
        dy = np.random.normal(0, self.step_size, size=(self.n_particles, self.n_steps))
        
        # 2. Cumulative sum to get positions
        x_trajectory = np.cumsum(dx, axis=1)
        y_trajectory = np.cumsum(dy, axis=1)
        
        # 3. Add the starting position (0,0) to the beginning of every array
        # This ensures t=0 is exactly at the origin
        # hstack effectively "glues" two arrays together side-by-side
        zeros = np.zeros((self.n_particles, 1))
        self.x_pos = np.hstack([zeros, x_trajectory])
        self.y_pos = np.hstack([zeros, y_trajectory])
        
        print(f"Simulation completed: {self.n_particles} particles over {self.n_steps} steps.")

    def get_std_deviation(self):
        """
        Calculates the standard deviation of distance from origin vs time.
        """
        if self.x_pos is None:
            raise ValueError("Run the simulation first!")
            
        # Calculate squared distance for every particle at every step
        r_squared = self.x_pos**2 + self.y_pos**2
        
        # Mean Squared Displacement (MSD) across all particles (axis=0)
        msd = np.mean(r_squared, axis=0)
        
        # Standard Deviation is sqrt(MSD)
        return np.sqrt(msd)

    def plot_results(self):
        """
        Plots the trajectories and the Standard Deviation analysis.
        """
        std_dev = self.get_std_deviation()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: The Cloud of Particles
        # We only plot the first 50 particles to keep the rendering fast and clean
        for i in range(min(50, self.n_particles)):
            ax1.plot(self.x_pos[i], self.y_pos[i], alpha=0.3, linewidth=1)
        
        ax1.plot(0, 0, 'ro', label='Start (0,0)', markersize=8)
        ax1.set_title(f"Particle Cloud (First 50 of {self.n_particles})")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Standard Deviation vs Time
        ax2.plot(self.time_axis, std_dev, 'b-', linewidth=2, label='Measured Std Dev')
        
        # We multiply by sqrt(2) because we have 2 dimensions contributing to the spread
        theory = self.step_size * np.sqrt(2 * self.time_axis)
        ax2.plot(self.time_axis, theory, 'r--', label=r'Theory ($\propto \sqrt{t}$)')
        
        ax2.set_title("Standard Deviation (Spread) vs Time")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Distance from Origin ($\sigma$)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

