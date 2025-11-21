import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
import os
import signal
import sys
from matplotlib.widgets import Button
import time
from up_mavt import mc_simulation, weighted_sum, geometric_mean, harmonic_mean, minimum, maximum

# ====================== DATA INPUT ======================
criteria = ["C1", "C2", "C3", "C4", "C5"]
alternative_A = [9, {"Normal": [7, 0.7]}, {"Discrete": [8, 8, 7]}, {"Uniform": [2, 3]}, {"Triangular": [1, 2, 3]}]
alternative_B = [6, {"Normal": [6, 0.6]}, {"Discrete": [7, 8, 6]}, {"Uniform": [1, 2]}, {"Triangular": [1, 1.5, 2]}]
alternative_C = [2, {"Normal": [3, 0.3]}, {"Discrete": [9, 6, 8]}, {"Uniform": [3, 6]}, {"Triangular": [2, 3, 4]}]
alternatives = [alternative_A, alternative_B, alternative_C]

# Value function points
C1_vf_a_points = [0, 3, 5, 6, 10]
C1_vf_b_points = [0, 2, 4, 7, 10]
C2_vf_a_points = [0, 4, 6, 8, 10]
C2_vf_b_points = [0, 3, 5, 7, 10]
C3_vf_a_points = [0, 2, 7, 9, 10]
C3_vf_b_points = [0, 1, 6, 8, 10]
C4_vf_a_points = [0, 1, 4, 6, 10]
C4_vf_b_points = [0, 2, 5, 7, 10]
C5_vf_a_points = [0, 2, 5, 8, 10]
C5_vf_b_points = [0, 1, 4, 7, 10]
elicitation_values = [0, 0.25, 0.5, 0.75, 1]

def value_function(*points):
    value_functions = []
    for point in points:
        vf = PchipInterpolator(point, elicitation_values)
        value_functions.append(vf)
    return value_functions

vf_C1 = value_function(C1_vf_a_points, C1_vf_b_points)
vf_C2 = value_function(C2_vf_a_points, C2_vf_b_points)
vf_C3 = value_function(C3_vf_a_points, C3_vf_b_points)
vf_C4 = value_function(C4_vf_a_points, C4_vf_b_points)
vf_C5 = value_function(C5_vf_a_points, C5_vf_b_points)
vf_list = [vf_C1, vf_C2, vf_C3, vf_C4, vf_C5]

# ====================== LIVE MONTE CARLO SIMULATION ======================
class LiveMCSimulation:
    def __init__(self, alternatives, vf_list, posterior_samples_list, aggregation_method=weighted_sum,
                 sim_runs=100000, weight_fixed=5, update_interval=100, strict=True, strict_d=True):
        """
        Live MC simulation wrapper that uses the existing mc_simulation function.

        Parameters:
        - update_interval: How often to update the plot (in iterations)
        - strict, strict_d: Passed to mc_simulation
        """
        self.alternatives = alternatives
        self.vf_list = vf_list
        self.posterior_samples_list = posterior_samples_list
        self.aggregation_method = aggregation_method
        self.sim_runs = sim_runs
        self.weight_fixed = weight_fixed
        self.update_interval = update_interval
        self.strict = strict
        self.strict_d = strict_d
        self.stop_simulation = False
        self.results = []
        self.relative_errors = []
        self.cumulative_means = []
        self.current_iteration = 0

        # Setup the figure
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        # Add stop button
        ax_stop = plt.axes([0.85, 0.01, 0.1, 0.05])
        self.stop_button = Button(ax_stop, 'Stop Simulation')
        self.stop_button.on_clicked(self.on_stop)

        # Create subplots
        self.ax_convergence = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
        self.ax_distribution = plt.subplot2grid((3, 2), (0, 1))
        self.ax_ranking = plt.subplot2grid((3, 2), (1, 1))
        self.ax_stats = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.02, 0.5, '', va='top', ha='left')

        # Initialize plot elements
        self.num_alternatives = len(alternatives)
        self.lines = []
        for alt_idx in range(self.num_alternatives):
            line, = self.ax_convergence.plot([], [], label=f'Alternative {chr(65+alt_idx)}')
            self.lines.append(line)
        self.ax_convergence.legend()
        self.ax_convergence.set_title('Convergence of Alternative Scores')
        self.ax_convergence.set_xlabel('Iteration')
        self.ax_convergence.set_ylabel('Cumulative Mean Score')
        self.ax_convergence.grid(True)

        plt.tight_layout()

    def on_stop(self, event):
        """Handle stop button click"""
        self.stop_simulation = True

    def on_close(self, event):
        """Handle figure close event"""
        self.stop_simulation = True

    def run_chunk(self, chunk_size):
        """Run a chunk of simulations and update plots"""
        # Run a chunk of simulations
        chunk_results, chunk_rel_errs = mc_simulation(
            self.alternatives, self.vf_list, self.posterior_samples_list,
            self.aggregation_method, sim_runs=chunk_size,
            weight_fixed=self.weight_fixed, strict=self.strict, strict_d=self.strict_d
        )

        # Process results
        for set_idx, result_set in enumerate(chunk_results):
            if set_idx >= len(self.results):
                self.results.append([])
                self.relative_errors.append([])
                self.cumulative_means.append([])

            self.results[set_idx].extend(result_set)
            self.relative_errors[set_idx].extend(chunk_rel_errs[set_idx])

            # Update cumulative means
            current_results = np.array(self.results[set_idx])
            current_means = np.mean(current_results, axis=0)
            self.cumulative_means[set_idx].append(current_means)

        self.current_iteration += chunk_size
        self.update_plot()

    def update_plot(self):
        """Update all plots with current results"""
        # Clear axes
        self.ax_convergence.clear()
        self.ax_distribution.clear()
        self.ax_ranking.clear()

        # Update convergence plot
        if self.cumulative_means:
            # Compute the average cumulative means across weight sets
            avg_means = np.mean([np.array(means) for means in self.cumulative_means if means], axis=0)
            for alt_idx in range(self.num_alternatives):
                self.ax_convergence.plot(
                    range(1, len(avg_means) + 1),
                    avg_means[:, alt_idx],
                    label=f'Alternative {chr(65 + alt_idx)}'
                )
        self.ax_convergence.set_title('Convergence of Alternative Scores (Averaged)')
        self.ax_convergence.set_xlabel('Iteration Chunk')
        self.ax_convergence.set_ylabel('Cumulative Mean Score')
        self.ax_convergence.legend()
        self.ax_convergence.grid(True)
        self.ax_convergence.set_xlim(0, (self.sim_runs // self.update_interval) + 1)

        # Update distribution plot (using last weight set)
        if self.results:
            last_results = np.array(self.results[-1])
            for alt_idx in range(self.num_alternatives):
                self.ax_distribution.hist(
                    last_results[:, alt_idx],
                    bins=20,
                    alpha=0.5,
                    label=f'Alternative {chr(65+alt_idx)}',
                    density=True
                )
        self.ax_distribution.set_title(f'Score Distributions (Iteration {self.current_iteration})')
        self.ax_distribution.legend()
        self.ax_distribution.grid(True)

        # Update ranking probabilities
        if self.results:
            last_results = np.array(self.results[-1])
            ranks = np.argsort(np.argsort(last_results, axis=1), axis=1) + 1
            ranking_probs = np.zeros((self.num_alternatives, self.num_alternatives))

            for alt_idx in range(self.num_alternatives):
                for rank in range(1, self.num_alternatives + 1):
                    ranking_probs[alt_idx, rank-1] = np.mean(ranks[:, alt_idx] == rank)

            self.ax_ranking.clear()
            im = self.ax_ranking.imshow(
            ranking_probs,
            cmap='YlGnBu',
            aspect='auto'
            )
            self.ax_ranking.set_xticks(range(self.num_alternatives))
            self.ax_ranking.set_xticklabels([f'A{chr(65+i)}' for i in range(self.num_alternatives)])
            self.ax_ranking.set_yticks(range(self.num_alternatives))
            self.ax_ranking.set_yticklabels([f'Rank {i+1}' for i in range(self.num_alternatives)])
            self.ax_ranking.set_title('Current Ranking Probabilities')

            # Ensure only one colorbar is present
            if not hasattr(self, 'ranking_colorbar'):
                self.ranking_colorbar = plt.colorbar(im, ax=self.ax_ranking)
            else:
                self.ranking_colorbar.update_normal(im)

        # Update stats
        stats = [
            f"Completed: {min(self.current_iteration, self.sim_runs)}/{self.sim_runs} iterations",
            f"Weight sets: {len(self.results)}/{self.weight_fixed if self.weight_fixed > 0 else 1}",
            f"Time: {time.strftime('%H:%M:%S')}"
        ]

        if self.results:
            last_results = np.array(self.results[-1])
            for alt_idx in range(self.num_alternatives):
                mean_val = np.mean(last_results[:, alt_idx])
                std_val = np.std(last_results[:, alt_idx])
                stats.append(f"Alt {chr(65+alt_idx)}: {mean_val:.4f} ± {std_val:.4f}")

        self.stats_text.set_text('\n'.join(stats))
        plt.pause(0.01)

    def run_simulation(self):
        """Run the complete simulation with live updates"""
        print(f"Starting simulation with {self.sim_runs} iterations...")
        print("Press the 'Stop Simulation' button or close the window to stop early.")

        try:
            while self.current_iteration < self.sim_runs and not self.stop_simulation:
                # Run in chunks for live updates
                chunk_size = min(self.update_interval, self.sim_runs - self.current_iteration)
                self.run_chunk(chunk_size)

            # If we stopped early, print message
            if self.stop_simulation:
                print(f"\nSimulation stopped at {self.current_iteration} iterations.")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")

        finally:
            # Finalize results
            plt.ioff()

            # Save results if we have any
            if self.results:
                timestamp = int(time.time())
                os.makedirs('results', exist_ok=True)

                # Combine all results
                final_results = []
                final_rel_errs = []
                final_cumulative = []

                for set_idx in range(len(self.results)):
                    if self.results[set_idx]:
                        final_results.append(np.array(self.results[set_idx]))
                        final_rel_errs.append(self.relative_errors[set_idx])
                        if self.cumulative_means[set_idx]:
                            final_cumulative.append(np.array(self.cumulative_means[set_idx]))

                np.save(f'results/mc_results_{timestamp}.npy', final_results)
                np.save(f'results/mc_relative_errors_{timestamp}.npy', final_rel_errs)
                np.save(f'results/mc_cumulative_means_{timestamp}.npy', final_cumulative)

                print(f"\nResults saved to 'results/mc_results_{timestamp}.npy'")
                print(f"Relative errors saved to 'results/mc_relative_errors_{timestamp}.npy'")
                print(f"Cumulative means saved to 'results/mc_cumulative_means_{timestamp}.npy'")

            plt.show()
            return final_results, final_rel_errs, final_cumulative

# Example usage
if __name__ == "__main__":
    # Load your data
    posterior_samples_list = [np.load("posterior_samples.npy")]

    # Create and run simulation
    sim = LiveMCSimulation(
        alternatives=alternatives,
        vf_list=vf_list,
        posterior_samples_list=posterior_samples_list,
        sim_runs=100000,
        weight_fixed=5,
        update_interval=1000,  # Update every 1000 iterations
        strict=True,
        strict_d=True
    )

    results, rel_errs, cumulative_means = sim.run_simulation()
