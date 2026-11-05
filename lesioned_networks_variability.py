#!/usr/bin/env python3
"""
Test variability of jPCA trajectories, neural activities, and movement trajectories
across repeated experiments for lesioned network (lesion size 0.1).
"""

import os
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from copy import deepcopy
import seaborn as sns

from utils import (load_trained_model, run_reaching_experiment, compute_jpca_trajectories)


class VariabilityTester:
    """Test variability across repeated experiments."""

    def __init__(self, model_name='test_0', n_runs=10, results_dir='variability_test'):
        self.model_name = model_name
        self.n_runs = n_runs
        directory = Path.cwd() / 'results' / model_name
        self.results_dir = Path(directory / results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Load the healthy model (will be used as base for creating lesioned copies)
        self.healthy_model, self.healthy_params = load_trained_model(model_name)

        # Parameters
        self.n_neurons = self.healthy_params['model']['n']
        self.tsteps = self.healthy_params['model']['tsteps']
        self.dt = self.healthy_params['model']['dt']

        # Store all run results
        self.all_runs_no_stim = {}
        self.all_runs_stim = {}

        # Current active runs
        self.all_runs = {}
        self.stimulation = False


    def run_single_experiment(self, lesion_fraction=0.1, stimulation_config=None):
        """
        Args:
            lesion_fraction: Fraction of neurons to lesion
            stimulation_config: Stimulation configuration
        """

        # Create a fresh copy of the healthy model
        model = deepcopy(self.healthy_model)

        # Apply lesion with random permutation (different each run)
        perm = torch.randperm(self.n_neurons)
        model.set_dead_fraction(perm, lesion_fraction)

        alive_count = model.alive_mask.sum().item()

        # Run experiment
        exp_results_s, exp_results = run_reaching_experiment(
            model, self.healthy_params,
            stimulation_config=stimulation_config,
            compare=True,
            fixed_movement_onset=False
        )

        return exp_results, exp_results_s


    def run_all_experiments(self, lesion_fraction=0.1, stimulation_config=None):

        for run_id in range(self.n_runs):
            results_no_stim, results_stim = self.run_single_experiment(lesion_fraction, stimulation_config)
            self.all_runs_no_stim[run_id] = results_no_stim
            self.all_runs_stim[run_id] = results_stim

        return self.all_runs_no_stim, self.all_runs_stim


    def compute_jpca_for_all_runs(self):

        jpca_results = {}
        for run_id, results in self.all_runs.items():
            activities = results['aligned_activities']
            projected, total_var, plane_variances, rot_features, _, jpcs_var = compute_jpca_trajectories(
                activities,
                n_jpcs=10,
                time_window=(0, 2.5),
                dt=self.dt
            )
            jpca_results[run_id] = {
                'projected': projected,
                'tids': results['tids'],
                'total_variance_explained': total_var,
                'jpcs_variances': jpcs_var,
                'plane_variances': plane_variances,
                'rotational_features': rot_features,
                'trajectory_metrics': results['trajectory_metrics'],
                'activities': activities
            }
        self.jpca_results = jpca_results

        return self.jpca_results


    def analyze_variability(self):
        """
        Analyze variability across runs.
        """
        variability_metrics = {}

        # 1. Trajectory metrics variability
        all_endpoint_errors = []
        all_straightness = []
        all_max_deviations = []
        all_total_var = []
        all_plane_variances = [[] for _ in range(5)]
        all_angular_velocities = [[] for _ in range(5)]
        all_rotational_frequencies = [[] for _ in range(5)]

        for run_id, results in self.jpca_results.items():
            tm = results['trajectory_metrics']
            all_endpoint_errors.append(tm['endpoint_error_mean'])
            all_straightness.append(tm['straightness_mean'])
            all_max_deviations.append(tm['max_deviation_mean'])
            all_total_var.append(results['total_variance_explained'])

            for p in range(5):
                all_plane_variances[p].append(results['plane_variances'][p])
                all_angular_velocities[p].append(
                    results['rotational_features'][f'plane_{p + 1}']['mean_angular_velocity']
                )
                all_rotational_frequencies[p].append(
                    results['rotational_features'][f'plane_{p + 1}']['mean_rotational_frequency']
                )

        variability_metrics = {
            'endpoint_error': {
                'mean': np.mean(all_endpoint_errors),
                'std': np.std(all_endpoint_errors),
                'cv': np.std(all_endpoint_errors) / np.mean(all_endpoint_errors) if np.mean(
                    all_endpoint_errors) > 0 else 0,
                'values': all_endpoint_errors
            },
            'straightness': {
                'mean': np.mean(all_straightness),
                'std': np.std(all_straightness),
                'cv': np.std(all_straightness) / np.mean(all_straightness) if np.mean(all_straightness) > 0 else 0,
                'values': all_straightness
            },
            'max_deviation': {
                'mean': np.mean(all_max_deviations),
                'std': np.std(all_max_deviations),
                'cv': np.std(all_max_deviations) / np.mean(all_max_deviations) if np.mean(
                    all_max_deviations) > 0 else 0,
                'values': all_max_deviations
            },
            'total_variance': {
                'mean': np.mean(all_total_var),
                'std': np.std(all_total_var),
                'cv': np.std(all_total_var) / np.mean(all_total_var) if np.mean(all_total_var) > 0 else 0,
                'values': all_total_var
            },
            'plane_variances': {},
            'angular_velocities': {},
            'rotational_frequencies': {}
        }

        for p in range(5):
            variability_metrics['plane_variances'][p] = {
                'mean': np.mean(all_plane_variances[p]),
                'std': np.std(all_plane_variances[p]),
                'cv': np.std(all_plane_variances[p]) / np.mean(all_plane_variances[p]) if np.mean(
                    all_plane_variances[p]) > 0 else 0,
                'values': all_plane_variances[p]
            }
            variability_metrics['angular_velocities'][p] = {
                'mean': np.mean(all_angular_velocities[p]),
                'std': np.std(all_angular_velocities[p]),
                'cv': np.std(all_angular_velocities[p]) / np.mean(all_angular_velocities[p]) if np.mean(
                    all_angular_velocities[p]) > 0 else 0,
                'values': all_angular_velocities[p]
            }
            variability_metrics['rotational_frequencies'][p] = {
                'mean': np.mean(all_rotational_frequencies[p]),
                'std': np.std(all_rotational_frequencies[p]),
                'cv': np.std(all_rotational_frequencies[p]) / np.mean(all_rotational_frequencies[p]) if np.mean(
                    all_rotational_frequencies[p]) > 0 else 0,
                'values': all_rotational_frequencies[p]
            }

        self.variability_metrics = variability_metrics


        return variability_metrics

    def plot_variability(self):
        """
        Create comprehensive variability plots.
        """
        if not hasattr(self, 'variability_metrics'):
            print("Please run analyze_variability() first!")
            return

        # Colors for different runs
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_runs))

        # =========================================================
        # PART 1: Movement Trajectories - One subplot per run
        # =========================================================
        n_cols = 5
        n_rows = (self.n_runs + n_cols - 1) // n_cols
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes1 = axes1.flatten()

        for run_id in range(self.n_runs):
            ax = axes1[run_id]
            per_target = self.all_runs[run_id]['per_target_results']

            # Plot trajectories for all targets (using target colors)
            target_colors = plt.cm.tab10(np.linspace(0, 1, 8))
            for target_id in range(8):
                if target_id in per_target:
                    mean_traj = per_target[target_id]['mean_trajectory']
                    target_pos = per_target[target_id]['target_position']

                    # Plot mean trajectory
                    ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                            color=target_colors[target_id], linewidth=1.5, alpha=0.7)

                    # Plot target position
                    ax.scatter(target_pos[0], target_pos[1],
                               color=target_colors[target_id], s=50, marker='o', linewidth=0.5)

            # Plot start position
            ax.scatter(0, 0, color='black', s=60, marker='o', zorder=10)

            # Add performance text
            tm = self.jpca_results[run_id]['trajectory_metrics']
            ax.text(0.05, 0.95, f'Error: {tm["endpoint_error_mean"]:.2f} cm',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)

            ax.set_xlabel('X (cm)')
            if run_id % n_cols == 0:
                ax.set_ylabel('Y (cm)')
            ax.set_title(f'Run {run_id}')
            ax.set_aspect('equal')
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-5, 0, 5])

        # Hide unused subplots
        for idx in range(self.n_runs, len(axes1)):
            axes1[idx].axis('off')

        plt.suptitle(f'Movement Trajectories Across {self.n_runs} Runs', fontsize=14)
        plt.tight_layout()
        if self.stimulation:
            plt.savefig(self.results_dir / 'variability_trajectories_s.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'variability_trajectories.png', dpi=150, bbox_inches='tight')
        plt.close()

        # =========================================================
        # PART 2: jPCA Trajectories - One subplot per plane per run
        # =========================================================
        n_planes = 5
        fig2, axes2 = plt.subplots(self.n_runs, n_planes,
                                   figsize=(5 * n_planes, 5 * self.n_runs))

        # Handle single run case
        if self.n_runs == 1:
            axes2 = axes2.reshape(1, -1)

        target_colors = plt.cm.tab10(np.linspace(0, 1, 8))

        for run_id in range(self.n_runs):
            result = self.jpca_results[run_id]
            projected = result['projected']
            tids = result['tids']
            plane_variances = result['plane_variances']

            # Group trials by target
            trials_by_target = {}
            for trial_idx, target_id in enumerate(tids):
                if target_id not in trials_by_target:
                    trials_by_target[target_id] = []
                trials_by_target[target_id].append(projected[trial_idx])

            for plane_idx in range(n_planes):
                ax = axes2[run_id, plane_idx]

                # Get indices for this plane
                idx1 = plane_idx * 2
                idx2 = plane_idx * 2 + 1

                # Plot each target separately
                for target_id in range(8):
                    if target_id not in trials_by_target:
                        continue

                    target_trials = trials_by_target[target_id]

                    # Calculate mean trajectory for this target
                    all_trajs = np.array([p[:, [idx1, idx2]] for p in target_trials])
                    mean_traj = np.mean(all_trajs, axis=0)

                    # Plot mean trajectory
                    ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                            color=target_colors[target_id], linewidth=1.5, alpha=0.8)

                    # Mark start and end
                    ax.scatter(mean_traj[0, 0], mean_traj[0, 1],
                               color=target_colors[target_id], s=30, marker='o',
                               alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1],
                               color=target_colors[target_id], s=30, marker='X',
                               alpha=0.7, edgecolor='black', linewidth=0.5)

                # Set labels
                ax.set_xlabel(f'jPC{idx1 + 1}')
                ax.set_ylabel(f'jPC{idx2 + 1}')

                # Title with plane info
                ax.set_title(f'Var: {plane_variances[plane_idx]:.3f}', fontsize=9)

                ax.set_aspect('equal')

                # Set consistent limits
                max_val = 2.0
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)

                # Set ticks
                ax.set_xticks([-2, -1, 0, 1, 2])
                ax.set_yticks([-2, -1, 0, 1, 2])

            if run_id == 0:
                # Add legend for each target
                handles = [plt.Line2D([0], [0], color=target_colors[target_id], lw=2)
                    for target_id in range(8)
                ]
                labels = [f"target {target_id}" for target_id in range(8)]
                fig2.legend(
                    handles, labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1),
                    ncol=len(labels),
                    fontsize=10
                )

        # Add row labels (run numbers)
        for run_id in range(self.n_runs):
            axes2[run_id, 0].set_ylabel(f'Run {run_id}', fontsize=10, labelpad=10)

        plt.suptitle(f'jPCA Trajectories Across {self.n_runs} Runs', fontsize=14, y=1.02)
        plt.tight_layout()
        if self.stimulation:
            plt.savefig(self.results_dir / 'variability_jpca_s.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'variability_jpca.png', dpi=150, bbox_inches='tight')
        plt.close()

        # =========================================================
        # PART 3: Summary statistics (boxplots)
        # =========================================================
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Behavioral metrics boxplot
        ax1 = axes3[0]
        metrics_names = ['Endpoint Error', 'Straightness', 'Max Deviation']
        metrics_values = [
            self.variability_metrics['endpoint_error']['values'],
            self.variability_metrics['straightness']['values'],
            self.variability_metrics['max_deviation']['values']
        ]
        bp1 = ax1.boxplot(metrics_values, tick_labels=metrics_names, patch_artist=True)
        for patch, color in zip(bp1['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        ax1.set_ylabel('Value')
        ax1.set_title('Behavioral Metrics Across Runs')
        ax1.grid(True, alpha=0.3)

        # 2. Total variance across runs
        ax2 = axes3[1]
        total_vars = self.variability_metrics['total_variance']['values']
        colors_run = plt.cm.tab10(np.linspace(0, 1, self.n_runs))
        ax2.bar(range(self.n_runs), total_vars, color=colors_run, alpha=0.7)
        ax2.axhline(self.variability_metrics['total_variance']['mean'], color='red', linestyle='--',
                    label=f"Mean: {self.variability_metrics['total_variance']['mean']:.3f}")
        ax2.fill_between(range(self.n_runs),
                         [self.variability_metrics['total_variance']['mean'] -
                          self.variability_metrics['total_variance']['std']] * self.n_runs,
                         [self.variability_metrics['total_variance']['mean'] +
                          self.variability_metrics['total_variance']['std']] * self.n_runs,
                         alpha=0.2, color='red')
        ax2.set_xlabel('Run')
        ax2.set_ylabel('Total Variance Explained')
        ax2.set_title('jPCA Total Variance Across Runs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Plane variances boxplot
        ax3 = axes3[2]
        plane_data = [self.variability_metrics['plane_variances'][p]['values'] for p in range(5)]
        bp3 = ax3.boxplot(plane_data, tick_labels=[f'Plane {p + 1}' for p in range(5)], patch_artist=True)
        for patch in bp3['boxes']:
            patch.set_facecolor('lightyellow')
        ax3.set_ylabel('Variance Explained')
        ax3.set_title('Plane Variances Across Runs')
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f'Variability Summary Across {self.n_runs} Runs', fontsize=14)
        plt.tight_layout()
        if self.stimulation:
            plt.savefig(self.results_dir / 'variability_summary_s.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'variability_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        return fig1, fig2, fig3


    def plot_all_neural_activity_difference(self, trial_idx=5, time_window=(0, 2.5), n_cols=5):
        """
        Plot neural activity difference for ALL runs in a grid layout (in original run order).

        Args:
            trial_idx: Trial index to plot (default 5)
            time_window: (start_time, end_time) in seconds
            n_cols: Number of columns in the grid
        """
        dt = self.dt
        start_idx = int(time_window[0] / dt)
        end_idx = int(time_window[1] / dt)

        n_rows = (self.n_runs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        # Get total variances (for title only, not for sorting)
        total_vars = [self.jpca_results[run_id]['total_variance_explained'] for run_id in range(self.n_runs)]

        # Use original run order (0, 1, 2, ...) instead of sorted
        for idx, run_id in enumerate(range(self.n_runs)):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get activity and compute difference
            activity = self.all_runs[run_id]['aligned_activities'][trial_idx, start_idx:end_idx, :]
            activity_diff = np.diff(activity, axis=0)
            activity_diff = np.vstack([np.zeros((1, activity_diff.shape[1])), activity_diff])

            times = np.arange(activity.shape[0]) * dt + time_window[0]

            # Plot activity difference
            vmax = 0.6
            im = ax.imshow(activity_diff.T, aspect='auto', cmap='seismic',
                           extent=[times[0], times[-1], 0, activity_diff.shape[1]],
                           origin='lower', vmin=-vmax, vmax=vmax)

            # Add stimulation marker if available
            stim_config = self.all_runs[run_id].get('stimulation_config')
            if stim_config is not None and stim_config.get('amplitude', 0) != 0:
                if stim_config['reference_time'] == 'TrialOnset':
                    stim_start = stim_config.get('start_time')
                elif stim_config['reference_time'] == 'MovementOnset':
                    stim_start = stim_config.get('start_time') + 1.2
                stim_duration = stim_config.get('duration')

                if stim_start is not None and stim_duration is not None:
                    ax.axvline(x=stim_start, color='cyan', linestyle='--', linewidth=1, alpha=0.8)
                    ax.axvline(x=stim_start + stim_duration, color='cyan', linestyle='--', linewidth=1, alpha=0.8)

            ax.set_xlabel('Time (s)')
            if idx % n_cols == 0:
                ax.set_ylabel('Neuron Index')
            ax.set_title(f'Run {run_id}\nVar: {total_vars[run_id]:.3f}')
            plt.colorbar(im, ax=ax, label='ΔActivity')

        # Hide unused subplots
        for idx in range(self.n_runs, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Neural Activity Difference (ΔActivity) Across {self.n_runs} Runs', fontsize=14)
        plt.tight_layout()
        if self.stimulation:
            plt.savefig(self.results_dir / 'all_neural_activity_difference_s.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'all_neural_activity_difference.png', dpi=150, bbox_inches='tight')
        plt.close()

        return fig


    def save_results(self):
        """Save all results to file."""
        all_data = {
            'n_runs': self.n_runs,
            'variability_metrics': self.variability_metrics,
            'jpca_results': {str(k): {
                'total_variance_explained': v['total_variance_explained'],
                'plane_variances': v['plane_variances']
            } for k, v in self.jpca_results.items()},
            'trajectory_metrics': {str(k): v['trajectory_metrics'] for k, v in self.jpca_results.items()}
        }
        if self.stimulation:
            with open(self.results_dir / 'variability_results_s.pkl', 'wb') as f:
                pickle.dump(all_data, f)
        else:
            with open(self.results_dir / 'variability_results.pkl', 'wb') as f:
                pickle.dump(all_data, f)


    def plot_cumulative_variance(self):
        """
        Plot cumulative explained variance by jPCs across all runs.
        X-axis: number of jPCs (1 to 10)
        Y-axis: cumulative explained variance
        Shows n_runs lines for the current stimulation condition.

        Uses 'jpcs_variances' from jpca_results (variances for each individual jPC).
        """
        if not hasattr(self, 'jpca_results'):
            print("Please run compute_jpca_for_all_runs() first!")
            return None

        n_jpcs = 10

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define colors for different runs
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_runs))

        for run_id in range(self.n_runs):
            # Get variances for each jPC
            jpcs_var = self.jpca_results[run_id]['jpcs_variances']
            cumulative = np.cumsum(jpcs_var)

            # Plot each run as a separate line
            x = np.arange(1, n_jpcs + 1)
            y = cumulative
            ax.plot(x, y, color=colors[run_id], linewidth=1.5, alpha=0.7,
                    label=f'Run {run_id}' if run_id < 10 else None)

        # Formatting
        ax.set_xlabel('Number of jPCs', fontsize=12)
        ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax.set_title(f'Cumulative jPCA Variance Across {self.n_runs} Runs\n'
                     f'({"With Stimulation" if self.stimulation else "Without Stimulation"})',
                     fontsize=14)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8, ncol=2)

        # Add text box with statistics
        stats_text = f"Mean total variance: {self.variability_metrics['total_variance']['mean']:.1f} ± {self.variability_metrics['total_variance']['std']:.3f}\n"

        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save figure
        if self.stimulation:
            plt.savefig(self.results_dir / 'cumulative_variance_s.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'cumulative_variance.png', dpi=150, bbox_inches='tight')
        plt.close()

        return fig


    def run_full_variability_test(self, lesion_fraction=0.1, stimulation_config=None):
        """Run complete variability test pipeline for BOTH conditions."""

        # Run all experiments (stores both no_stim and stim results)
        self.run_all_experiments(lesion_fraction=lesion_fraction, stimulation_config=stimulation_config)

        # =========================================================
        # CONDITION 1: WITHOUT STIMULATION
        # =========================================================
        self.all_runs = self.all_runs_no_stim  # Use the no_stim results
        self.stimulation = False

        # Compute jPCA
        self.compute_jpca_for_all_runs()

        # Analyze variability
        self.analyze_variability()

        # Plot variability
        self.plot_variability()
        self.plot_all_neural_activity_difference()
        self.plot_cumulative_variance()

        # Save results
        self.save_results()

        # =========================================================
        # CONDITION 2: WITH STIMULATION
        # =========================================================
        self.all_runs = self.all_runs_stim  # Use the stim results
        self.stimulation = True

        # Compute jPCA
        self.compute_jpca_for_all_runs()

        # Analyze variability
        self.analyze_variability()

        # Plot variability
        self.plot_variability()
        self.plot_all_neural_activity_difference()
        self.plot_cumulative_variance()

        # Save results
        self.save_results()

        return 0

def main():

    t = 0

    stimulation_config = {
        'ID': '0',
        'sites': [(0.5, 0.5)],
        'transform': 'reciprocal',
        'combine': 'sum',
        'transform_params': {'max_distance': 0.03, 'power': 1},
        'reference_time': 'MovementOnset',  # TODO ==== Check the stimulation reference time ====
        'amplitude': 3,
        'stim_type': 'pulse',
        'start_time': t,
        'duration': 0.2
    }

    tester = VariabilityTester(
        model_name='test_0',
        n_runs=10,  # Number of repeated experiments
        results_dir='variability_test_t'+str(t)+ '+amp3'
    )
    tester.run_full_variability_test(lesion_fraction=0.1, stimulation_config=stimulation_config)


if __name__ == "__main__":
    main()