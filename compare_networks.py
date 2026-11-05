#!/usr/bin/env python3
"""
Compare healthy vs lesioned networks with/without stimulation.
Analyzes behavior (trajectories, accuracy) and neural dynamics (jPCA).
"""
import os
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.interpolate import interp1d
import pickle
from copy import deepcopy

from utils import (load_trained_model, compute_jpca_trajectories, run_reaching_experiment)
from generate_lesion_permutations import load_lesion_permutations

class NetworkComparator:

    def __init__(self, model_name, results_dir='test_name', fixed_movement_onset=False):

        self.fixed_movement_onset = fixed_movement_onset
        directory = Path.cwd() / 'results'/ model_name
        self.results_dir = Path(directory / results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Load the healthy model
        self.healthy_model, self.healthy_params = load_trained_model(model_name)

        # Parameters
        self.n_neurons = self.healthy_params['model']['n']
        self.tsteps = self.healthy_params['model']['tsteps']
        self.dt = self.healthy_params['model']['dt']

    def run_all_conditions(self, stim_config, fixed_perm=None):
        """
        Run all conditions (healthy and lesioned) with optional fixed lesion permutation.

        Args:
            stim_config: Stimulation configuration dictionary
            fixed_perm: Pre-generated permutation tensor for lesion (if None, generates random)

        Returns:
            results_stim: Dictionary with results for conditions WITH stimulation
            results_nostim: Dictionary with results for conditions WITHOUT stimulation
        """
        import copy

        results_stim = {}
        results_nostim = {}
        conditions = [
            {'name': 'Healthy Network', 'lesion_fraction': 0},
            {'name': 'Lesioned Network', 'lesion_fraction': 0.1}
        ]

        for cond in conditions:
            model = copy.deepcopy(self.healthy_model)
            p_dead = cond['lesion_fraction']

            if p_dead > 0:
                if fixed_perm is not None:
                    # Use the provided fixed permutation
                    model.set_dead_fraction(fixed_perm, p_dead)
                else:
                    # Generate a new random permutation
                    perm = torch.randperm(self.n_neurons)
                    model.set_dead_fraction(perm, p_dead)

            # Run experiment with stimulation
            exp_results_stim, exp_results_nostim = run_reaching_experiment(
                model, self.healthy_params,
                stimulation_config=stim_config,
                compare=True,
                fixed_movement_onset=self.fixed_movement_onset
            )

            # Store results
            def store_results(results_dict, exp_results, cond_name):
                results_dict[cond_name] = {
                    'activities': exp_results['activities'],
                    'aligned_activities': exp_results['aligned_activities'],
                    'hidden_state': exp_results['hidden_state'],
                    'aligned_hidden': exp_results['aligned_hidden'],
                    'outputs': exp_results['outputs'],
                    'positions': exp_results['positions'],
                    'tids': exp_results['tids'],
                    'trajectory_metrics': exp_results['trajectory_metrics'],
                    'lesion_fraction': cond['lesion_fraction'],
                    'stimulation_config': exp_results['stimulation_config'],
                    'stimulation_weights': exp_results['stimulation_weights'],
                    'stimulation_intensity': exp_results['stimulation_intensity'],
                    'per_target_results': exp_results.get('per_target_results', {})
                }

            store_results(results_stim, exp_results_stim, cond['name'])
            store_results(results_nostim, exp_results_nostim, cond['name'])

        return results_stim, results_nostim

    def get_sorted_neurons_by_distance(self, site=(0.5, 0.5)):
        """Return neuron indices and distances sorted by distance from stimulation site."""
        positions = self.healthy_model.neuron_positions.cpu().numpy()
        distances = np.sqrt((positions[:, 0] - site[0]) ** 2 + (positions[:, 1] - site[1]) ** 2)
        sorted_indices = np.argsort(distances)
        return sorted_indices, distances[sorted_indices]

    def _compute_peak_responses(self, activities, activities_c, hidden, hidden_c,
                                stim_time, amplitude):
        """Compute peak responses for all neurons."""
        dt = self.dt  # Use instance's dt
        stim_idx = int(stim_time / dt)
        window_end = min(activities.shape[1], stim_idx + int(0.3 / dt))

        n_neurons = activities.shape[2]
        activity_peaks = np.zeros(n_neurons)
        hidden_peaks = np.zeros(n_neurons)

        for neuron_i in range(n_neurons):
            activity_trials = []
            hidden_trials = []

            for trial_idx in range(activities.shape[0]):
                # Activity peak
                diff = (activities[trial_idx, :, neuron_i] -
                        activities_c[trial_idx, :, neuron_i])
                peak = np.max(diff[stim_idx:window_end]) if amplitude >= 0 else np.min(diff[stim_idx:window_end])
                activity_trials.append(peak)

                # Hidden peak
                diff = (hidden[trial_idx, :, neuron_i] -
                        hidden_c[trial_idx, :, neuron_i])
                peak = np.max(diff[stim_idx:window_end]) if amplitude >= 0 else np.min(diff[stim_idx:window_end])
                hidden_trials.append(peak)

            activity_peaks[neuron_i] = np.mean(activity_trials)
            hidden_peaks[neuron_i] = np.mean(hidden_trials)

        return activity_peaks, hidden_peaks

    def plot_stimulation_characterization(self, results_stim, results_nostim,
                                          amplitude, start_time, save_path=None):
        """
        Plot stimulation characterization: heatmaps, distance response, intensity-response.
        Uses the lesioned network data from results_stim and results_nostim.
        """
        # Extract data
        data_stim = results_stim['Lesioned Network']
        data_nostim = results_nostim['Lesioned Network']

        aligned_act = data_stim['aligned_activities']
        aligned_act_c = data_nostim['aligned_activities']
        aligned_hid = data_stim['aligned_hidden']
        aligned_hid_c = data_nostim['aligned_hidden']

        plot_start_time = start_time + 1.2  # Convert to aligned time
        stim_end_time = plot_start_time + 0.2  # duration = 200ms

        # Get stimulation weights for sorting
        stim_weights = data_stim.get('stimulation_weights')
        if stim_weights is not None:
            stim_weights = stim_weights.cpu().numpy()
            sorted_idx = np.argsort(stim_weights)
        else:
            sorted_idx = np.arange(self.n_neurons)

        times = np.arange(aligned_act.shape[1]) * self.dt

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Neural Activity heatmap
        diff = np.mean(aligned_act, axis=0) - np.mean(aligned_act_c, axis=0)
        vmax = np.max(np.abs(diff))
        im = axes[0].imshow(diff[:, sorted_idx].T, aspect='auto', cmap='bwr',
                            extent=[times[0], times[-1], 0, self.n_neurons],
                            origin='lower', vmin=-vmax, vmax=vmax)
        axes[0].set_title(f'Neural Activity Difference (Stim - No Stim)\nAmplitude={amplitude}, t={start_time}s')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Neuron Index')
        plt.colorbar(im, ax=axes[0], label='ΔNeural Activity')
        axes[0].axvline(plot_start_time, color='cyan', linestyle='--', alpha=0.8, label='Stim ON')
        axes[0].axvline(stim_end_time, color='cyan', linestyle='--', alpha=0.8, label='Stim OFF')

        # Activation State heatmap
        diff = np.mean(aligned_hid, axis=0) - np.mean(aligned_hid_c, axis=0)
        vmax = np.max(np.abs(diff))
        im = axes[1].imshow(diff[:, sorted_idx].T, aspect='auto', cmap='bwr',
                            extent=[times[0], times[-1], 0, self.n_neurons],
                            origin='lower', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f'Activation State Difference (Stim - No Stim)\nAmplitude={amplitude}, t={start_time}s')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Neuron Index')
        plt.colorbar(im, ax=axes[1], label='ΔActivation State')
        axes[1].axvline(plot_start_time, color='cyan', linestyle='--', alpha=0.8)
        axes[1].axvline(stim_end_time, color='cyan', linestyle='--', alpha=0.8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_distance_response(self, results_stim, results_nostim, amplitude, start_time, save_path=None):
        """
        Plot distance-dependent response for a SINGLE run.

        Args:
            results_stim: results dictionary from run_all_conditions (with stimulation)
            results_nostim: results dictionary from run_all_conditions (without stimulation)
            amplitude: Stimulation amplitude (for title)
            start_time: Stimulation start time (for title)
            save_path: Path to save the figure (optional)
        """
        # Extract lesioned network data
        data_stim = results_stim['Lesioned Network']
        data_nostim = results_nostim['Lesioned Network']

        # Get sorted neurons by distance
        sorted_indices, sorted_distances = self.get_sorted_neurons_by_distance()

        # Compute stimulation time in aligned coordinates
        plot_start_time = start_time + 1.2

        # Compute peak responses
        activity_peaks, hidden_peaks = self._compute_peak_responses(
            data_stim['aligned_activities'], data_nostim['aligned_activities'],
            data_stim['aligned_hidden'], data_nostim['aligned_hidden'],
            plot_start_time, amplitude
        )

        # Sort by distance
        activity_sorted = activity_peaks[sorted_indices]
        hidden_sorted = hidden_peaks[sorted_indices]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Activity scatter
        ax = axes[0, 0]
        ax.scatter(sorted_distances, activity_sorted, alpha=0.3, s=5, color='steelblue')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Peak Response')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Activity binned
        ax = axes[0, 1]
        n_bins = 20
        bin_edges = np.linspace(0, np.max(sorted_distances), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        binned_means, binned_stds = [], []
        for i in range(n_bins):
            mask = (sorted_distances >= bin_edges[i]) & (sorted_distances < bin_edges[i + 1])
            if np.any(mask):
                binned_means.append(np.mean(activity_sorted[mask]))
                binned_stds.append(np.std(activity_sorted[mask]))
            else:
                binned_means.append(np.nan)
                binned_stds.append(np.nan)

        ax.errorbar(bin_centers, binned_means, yerr=binned_stds,
                    marker='o', capsize=5, markersize=6, linewidth=2, color='steelblue')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Peak Response')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Hidden scatter
        ax = axes[1, 0]
        ax.scatter(sorted_distances, hidden_sorted, alpha=0.3, s=5, color='coral')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Peak Response')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Hidden binned
        ax = axes[1, 1]
        binned_means, binned_stds = [], []
        for i in range(n_bins):
            mask = (sorted_distances >= bin_edges[i]) & (sorted_distances < bin_edges[i + 1])
            if np.any(mask):
                binned_means.append(np.mean(hidden_sorted[mask]))
                binned_stds.append(np.std(hidden_sorted[mask]))
            else:
                binned_means.append(np.nan)
                binned_stds.append(np.nan)

        ax.errorbar(bin_centers, binned_means, yerr=binned_stds,
                    marker='o', capsize=5, markersize=6, linewidth=2, color='coral')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Peak Response')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Distance-Dependent Stimulation Response\nAmplitude={amplitude}, t={start_time}s', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_neural_activity(self, results, condition_name=None, stim='with Stimulation', idx=5):
        """
        Plot neural activity and activity differences, with optional sorting by distance to stimulation site.

        Args:
            results: results dictionary from run_all_conditions (contains per_target_results or full results)
            condition_name: Name of the condition to plot (if None, uses first condition)
            stim: could be 'with Stimulation' or 'without Stimulation'. It is for titles of the plots.
            idx: Trial index to plot (default 5)
        """
        # Select condition
        if condition_name is None:
            condition_name = list(results.keys())[0]

        data = results[condition_name]

        if self.fixed_movement_onset:
            activity = data['activities']  # [n_trials, time, neurons]
            time_window = (0, 3)
        else:
            activity = data['aligned_activities']
            time_window = (0, 2.5)

        # Extract time window
        dt = 0.01
        start_idx = int(time_window[0] / dt)
        end_idx = int(time_window[1] / dt)

        # Get single trial activity
        if idx >= activity.shape[0]:
            idx = 0
        trial_activity = activity[idx, start_idx:end_idx, :]  # [time, neurons]

        # Get activity difference (difference between consecutive time steps)
        activity_diff = np.diff(trial_activity, axis=0)  # [time-1, neurons]
        # Pad to same length for plotting (add zero at beginning)
        activity_diff = np.vstack([np.zeros((1, activity_diff.shape[1])), activity_diff])

        # Get time points
        times = np.arange(trial_activity.shape[0]) * dt + time_window[0]

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Set colobar limit
        vmax = 0.6
        vmax_a = 6

        def add_stimulation_marker(ax, times):
            if self.fixed_movement_onset:
                stim_start_time = data['stimulation_config']['start_time']
            else:
                stim_start_time = data['stimulation_config']['start_time'] + 1.2
            stim_end_time = stim_start_time + data['stimulation_config']['duration']

            if stim_start_time <= times[-1] and stim_end_time >= times[0]:
                # Add vertical lines at stimulation start and end
                ax.axvline(x=stim_start_time, color='cyan', linestyle='--',
                           linewidth=1.5, alpha=0.8, label='Stim ON')
                ax.axvline(x=stim_end_time, color='cyan', linestyle='--',
                           linewidth=1.5, alpha=0.8, label='Stim OFF')
                # Add shaded region for stimulation period
                # ax.axvspan(stim_start_time, stim_end_time, alpha=0.2, color='cyan', label='Stimulation')

        # =========================================================
        # Subplot 1: Neural activity (not sorted)
        # =========================================================
        ax1 = axes[0, 0]
        im1 = ax1.imshow(trial_activity.T, aspect='auto', cmap='viridis',
                         extent=[times[0], times[-1], 0, trial_activity.shape[1]],
                         origin='lower', vmin=0, vmax=vmax_a)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Neuron Index')
        plt.colorbar(im1, ax=ax1, label='Activity')

        # =========================================================
        # Subplot 2: Activity difference (not sorted)
        # =========================================================
        ax2 = axes[0, 1]
        im2 = ax2.imshow(activity_diff.T, aspect='auto', cmap='bwr',
                         extent=[times[0], times[-1], 0, activity_diff.shape[1]],
                         origin='lower', vmin=-vmax, vmax=vmax)  #vmin=-np.abs(activity_diff).max(), vmax=np.abs(activity_diff).max())
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Neuron Index')
        plt.colorbar(im2, ax=ax2, label='ΔActivity')

        # =========================================================
        # Sorting by distance to stimulation site
        # =========================================================
        stim_weights = data['stimulation_weights']
        if stim_weights is not None:
            stim_weights = stim_weights.cpu().numpy()
            sorted_indices = np.argsort(stim_weights)  # strongest starts from 400
            sorting_method = "Sorted by Stimulation Weight"
            # Sort activity by distance to stimulation site
            trial_activity_sorted = trial_activity[:, sorted_indices]
            activity_diff_sorted = activity_diff[:, sorted_indices]

            # =========================================================
            # Subplot 3: Neural activity (sorted)
            # =========================================================
            ax3 = axes[1, 0]
            im3 = ax3.imshow(trial_activity_sorted.T, aspect='auto', cmap='viridis',
                             extent=[times[0], times[-1], 0, trial_activity_sorted.shape[1]],
                             origin='lower', vmin=0, vmax=vmax_a)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Neuron Index (Sorted)')
            plt.colorbar(im3, ax=ax3, label='Activity')
            add_stimulation_marker(ax3, times)

            # =========================================================
            # Subplot 4: Activity difference (sorted)
            # =========================================================
            ax4 = axes[1, 1]
            im4 = ax4.imshow(activity_diff_sorted.T, aspect='auto', cmap='bwr',
                             extent=[times[0], times[-1], 0, activity_diff_sorted.shape[1]],
                             origin='lower', vmin=-vmax, vmax=vmax) #vmin=-np.abs(activity_diff).max(), vmax=np.abs(activity_diff).max())
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Neuron Index (Sorted)')
            plt.colorbar(im4, ax=ax4, label='ΔActivity')
            add_stimulation_marker(ax4, times)

        else:
            # If no sorting information available, show placeholder
            for ax in [axes[1, 0], axes[1, 1]]:
                ax.text(0.5, 0.5, 'Sorting not available\n(neuron positions or stimulation weights missing)',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Not Available')

        plt.suptitle(f'Neural Activity Analysis - {condition_name} {stim}, Trial {idx}', fontsize=14)
        plt.tight_layout()

        plt.savefig(self.results_dir / f'neural_activity_{condition_name}_{stim}.svg',
                    bbox_inches='tight')
        plt.close()

    def plot_trajectory_comparison(self, results, individual_trials=False, n_individual=5, plot_mode='line'):
        """
        Plot trajectories for each condition (all targets in one subplot).

        Args:
            results: results dictionary from run_all_conditions
            individual_trajectories: If True, show random individual trajectories
            n_individual: Number of individual trajectories to show per target (if individual_trajectories=True)
            plot_mode: 'line' for continuous lines, 'points' for scatter points showing time progression
        """
        conditions = list(results.keys())
        n_conds = len(conditions)

        # Create figure: one subplot per condition
        fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 5))
        if n_conds == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, 8))

        for cond_idx, cond_name in enumerate(conditions):
            ax = axes[cond_idx]
            per_target = results[cond_name]['per_target_results']
            ntargets = len(per_target)

            for target_id in range(ntargets):
                if target_id in per_target:
                    target_data = per_target[target_id]
                    mean_traj = target_data['mean_trajectory']
                    target_pos = target_data['target_position']

                    if plot_mode == 'line':
                        # Plot mean trajectory as continuous line
                        ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                                color=colors[target_id], linewidth=2)
                    elif plot_mode == 'points':
                        downsample_factor = 5
                        time_indices = np.arange(0, mean_traj.shape[0], downsample_factor)
                        ax.scatter(mean_traj[time_indices, 0], mean_traj[time_indices, 1],
                                   color=colors[target_id], s=15, alpha=0.7, zorder=5)

                    # Optionally plot individual trajectories
                    if individual_trials:
                        positions = target_data['positions']
                        n_trials = positions.shape[0]
                        random_subset = np.random.choice(n_trials, min(n_individual, n_trials), replace=False)
                        for trial_idx in random_subset:
                            ax.plot(positions[trial_idx, :, 0], positions[trial_idx, :, 1],
                                        color=colors[target_id], alpha=0.2, linewidth=0.8)

                    # Plot target position
                    ax.scatter(target_pos[0], target_pos[1],
                               color=colors[target_id], s=100, marker='o', zorder=10, linewidth=1,
                               edgecolor='black')

            # Plot start position
            ax.scatter(0, 0, color='black', s=100, marker='o', zorder=10)

            # Add performance text
            tm = results[cond_name]['trajectory_metrics']
            ax.text(0.05, 0.95, f'Error: {tm["endpoint_error_mean"]:.2f}±{tm["endpoint_error_std"]:.2f} cm\n'
                                f'Straight: {tm["straightness_mean"]:.2f}±{tm["straightness_std"]:.2f}\n'
                                f'Max Dev: {tm["max_deviation_mean"]:.2f}±{tm["max_deviation_std"]:.2f} cm',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)

            ax.set_xlabel('X Position (cm)')
            if cond_idx == 0:
                ax.set_ylabel('Y Position (cm)')
            ax.set_title(cond_name)
            ax.set_aspect('equal')
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-5, 0, 5])

        plt.suptitle('Movement Trajectory\n'
                     f'{"(With Individual Trajectories)" if individual_trials else "(Mean Trajectories Only)"}',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(
            self.results_dir / f'trajectory_comparison_{plot_mode}{"_individual" if individual_trials else ""}.svg',
            bbox_inches='tight')
        plt.close()

    def plot_jpca_comparison(self, jpca_results, individual_trials=False, n_individual=5, plot_mode='line'):
        """
        Plot jPCA trajectories for all 5 planes.
        Each condition gets a row, each plane gets a column.
        Shows mean trajectory for each target.

        Args:
            jpca_results: Dictionary with keys like 'Healthy_No_Stim', 'Healthy_Stim', etc.
            individual_trials: If True, show random individual trials for each target
            n_individual: Number of individual trials to show per target (if individual_trials=True)
            plot_mode: 'line' for continuous lines, 'points' for scatter points showing time progression
        """
        conditions = list(jpca_results.keys())
        n_conds = len(conditions)
        n_planes = 5

        # Get target colors
        target_colors = plt.cm.tab10(np.linspace(0, 1, 8))

        # Create figure: rows = conditions, columns = planes
        fig, axes = plt.subplots(n_conds, n_planes, figsize=(5 * n_planes, 5 * n_conds))

        # Handle single condition case
        if n_conds == 1:
            axes = axes.reshape(1, -1)

        for cond_idx, cond_name in enumerate(conditions):
            result = jpca_results[cond_name]
            projected = result['projected']
            tids = result['tids']
            plane_variances = result['plane_variances']
            rot_features = result['rotational_features']

            # Group trials by target
            trials_by_target = {}
            for trial_idx, target_id in enumerate(tids):
                if target_id not in trials_by_target:
                    trials_by_target[target_id] = []
                trials_by_target[target_id].append(projected[trial_idx])
            ntargets = len(trials_by_target)

            for plane_idx in range(n_planes):
                ax = axes[cond_idx, plane_idx]

                # Get indices for this plane
                idx1 = plane_idx * 2
                idx2 = plane_idx * 2 + 1

                # Plot each target separately
                for target_id in range(ntargets):
                    target_trials = trials_by_target[target_id]

                    # Calculate mean trajectory for this target
                    all_trajs = np.array([p[:, [idx1, idx2]] for p in target_trials])
                    mean_traj = np.mean(all_trajs, axis=0)

                    if plot_mode == 'line':
                        # Plot mean trajectory as continuous line
                        ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                                color=target_colors[target_id], linewidth=2)
                    elif plot_mode == 'points':
                        downsample_factor = 2
                        time_indices = np.arange(0, mean_traj.shape[0], downsample_factor)
                        ax.scatter(mean_traj[time_indices, 0], mean_traj[time_indices, 1],
                                   color=target_colors[target_id], s=20, alpha=0.8, zorder=5,
                                   edgecolor='black', linewidth=0.2)

                    # Mark start and end (always as points)
                    ax.scatter(mean_traj[0, 0], mean_traj[0, 1],
                               color=target_colors[target_id], s=50, marker='o',
                               alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1],
                               color=target_colors[target_id], s=50, marker='X',
                               alpha=0.7, edgecolor='black', linewidth=0.5)

                    # Optionally plot individual trials
                    if individual_trials and n_individual > 0:
                        n_trials_available = len(target_trials)
                        random_subset = np.random.choice(range(n_trials_available),
                                                         min(n_individual, n_trials_available),
                                                         replace=False)
                        for trial_idx in random_subset:
                            traj = target_trials[trial_idx][:, [idx1, idx2]]
                            ax.plot(traj[:, 0], traj[:, 1],
                                        color=target_colors[target_id],
                                        alpha=0.15, linewidth=0.6)

                # Set labels
                ax.set_xlabel(f'jPC{idx1 + 1}')
                ax.set_ylabel(f'jPC{idx2 + 1}')

                # Get rotational features for this plane
                rot_feat = rot_features[f'plane_{plane_idx + 1}']
                rot_freq = rot_feat['mean_rotational_frequency']

                # Title with plane info
                ax.set_title(f'Var: {plane_variances[plane_idx]:.1f} | ν: {rot_freq:.2f} Hz',
                             fontsize=9)

                ax.set_aspect('equal')

                # Set consistent limits
                max_val = 2.2
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)

                # Set ticks
                ax.set_xticks([-2, -1, 0, 1, 2])
                ax.set_yticks([-2, -1, 0, 1, 2])

        # Add overall legend for targets
        handles = [plt.Line2D([0], [0], color=target_colors[target_id], lw=2)
                   for target_id in range(8)]
        labels = [f"target {target_id+1}" for target_id in range(8)]
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            ncol=len(labels),
            fontsize=10
        )

        plt.suptitle(f'jPCA Trajectories: Rotational Dynamics',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        suffix = f"_{plot_mode}{'_individual' if individual_trials else ''}"
        plt.savefig(self.results_dir / f'jpca_comparison_all_planes{suffix}.svg',bbox_inches='tight')
        plt.close()

    def plot_multi_target_responses(self, results_stim, results_nostim, amplitude, start_time,
                                neuron_idx=None, neuron_description=None, save_path=None):
        """
        Plot multi-target responses (by-target differences and bar plots) for a SINGLE run.

        Args:
            results_stim: results dictionary from run_all_conditions (with stimulation)
            results_nostim: results dictionary from run_all_conditions (without stimulation)
            amplitude: Stimulation amplitude (for title)
            start_time: Stimulation start time (for title)
            neuron_idx: Neuron index to plot
            neuron_description: Description of neuron (e.g., 'Closest', '2nd Closest', 'Farthest')
            save_path: Path to save the figure (optional)
        """
        # Extract lesioned network data
        data_stim = results_stim['Lesioned Network']
        data_nostim = results_nostim['Lesioned Network']

        aligned_act = data_stim['aligned_activities']
        aligned_act_c = data_nostim['aligned_activities']
        aligned_hid = data_stim['aligned_hidden']
        aligned_hid_c = data_nostim['aligned_hidden']
        tids = data_stim['tids']

        plot_start_time = start_time + 1.2
        stim_end_time = plot_start_time + 0.2  # duration = 200ms
        unique_targets = np.unique(tids)
        times = np.arange(aligned_act.shape[1]) * self.dt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, 8))

        # Build title suffix
        if neuron_description is not None:
            title_suffix = f"{neuron_description} Neuron"
        else:
            title_suffix = f"Neuron {neuron_idx}"

        # Activity by target
        for idx, target_id in enumerate(sorted(unique_targets)):
            mask = tids == target_id
            mean_with = np.mean(aligned_act[mask], axis=0)[:, neuron_idx]
            mean_without = np.mean(aligned_act_c[mask], axis=0)[:, neuron_idx]
            axes[0, 0].plot(times, mean_with - mean_without, color=colors[idx], label=f'Target {target_id + 1}')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Δ Neural Activity')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        axes[0, 0].axvline(plot_start_time, color='cyan', linestyle='--', alpha=0.8, label='Stim ON')
        axes[0, 0].axvline(stim_end_time, color='cyan', linestyle='--', alpha=0.8, label='Stim OFF')
        axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].grid(True, alpha=0.3)

        # Activity peak bar plot
        peaks, stds = [], []
        stim_idx = int(plot_start_time / self.dt)
        window_end = stim_idx + 50
        for target_id in sorted(unique_targets):
            mask = tids == target_id
            mean_with = np.mean(aligned_act[mask], axis=0)[:, neuron_idx]
            mean_without = np.mean(aligned_act_c[mask], axis=0)[:, neuron_idx]
            diff = mean_with - mean_without
            if amplitude >= 0:
                peak = np.max(diff[stim_idx:window_end])
            else:
                peak = np.min(diff[stim_idx:window_end])
            peaks.append(peak)
            trial_peaks = []
            for t in np.where(mask)[0]:
                trial_diff = aligned_act[t, :, neuron_idx] - aligned_act_c[t, :, neuron_idx]
                if amplitude >= 0:
                    trial_peak = np.max(trial_diff[stim_idx:window_end])
                else:
                    trial_peak = np.min(trial_diff[stim_idx:window_end])
                trial_peaks.append(trial_peak)
            stds.append(np.std(trial_peaks))

        axes[0, 1].bar([f'{t + 1}' for t in range(8)], peaks, yerr=stds,
                       capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Target')
        axes[0, 1].set_ylabel('Peak Response')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x')

        # Hidden by target
        for idx, target_id in enumerate(sorted(unique_targets)):
            mask = tids == target_id
            mean_with = np.mean(aligned_hid[mask], axis=0)[:, neuron_idx]
            mean_without = np.mean(aligned_hid_c[mask], axis=0)[:, neuron_idx]
            axes[1, 0].plot(times, mean_with - mean_without, color=colors[idx], label=f'Target {target_id + 1}')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Δ Activation State')
        axes[1, 0].axvline(plot_start_time, color='cyan', linestyle='--', alpha=0.8)
        axes[1, 0].axvline(stim_end_time, color='cyan', linestyle='--', alpha=0.8)
        axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # Hidden peak bar plot
        peaks, stds = [], []
        for target_id in sorted(unique_targets):
            mask = tids == target_id
            mean_with = np.mean(aligned_hid[mask], axis=0)[:, neuron_idx]
            mean_without = np.mean(aligned_hid_c[mask], axis=0)[:, neuron_idx]
            diff = mean_with - mean_without
            if amplitude >= 0:
                peak = np.max(diff[stim_idx:window_end])
            else:
                peak = np.min(diff[stim_idx:window_end])
            peaks.append(peak)
            trial_peaks = []
            for t in np.where(mask)[0]:
                trial_diff = aligned_hid[t, :, neuron_idx] - aligned_hid_c[t, :, neuron_idx]
                if amplitude >= 0:
                    trial_peak = np.max(trial_diff[stim_idx:window_end])
                else:
                    trial_peak = np.min(trial_diff[stim_idx:window_end])
                trial_peaks.append(trial_peak)
            stds.append(np.std(trial_peaks))

        axes[1, 1].bar([f'{t + 1}' for t in range(8)], peaks, yerr=stds,
                       capsize=5, color='coral', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Target')
        axes[1, 1].set_ylabel('Peak Response')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x')

        plt.suptitle(f'Multi-Target Responses - Amp={amplitude}, t={start_time}s', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def explore_single_neuron_response(self, results_stim, results_nostim, amplitude, start_time,
                                       neuron_idx=None, neuron_description=None, save_dir=None):
        """
        Explore single neuron response (activity and hidden state traces) for a SINGLE run.

        Args:
            results_stim: results dictionary from run_all_conditions (with stimulation)
            results_nostim: results dictionary from run_all_conditions (without stimulation)
            amplitude: Stimulation amplitude (for title)
            start_time: Stimulation start time (for title)
            neuron_idx: Neuron index to plot
            neuron_description: Description of neuron (e.g., 'Closest', '2nd Closest', 'Farthest')
            save_dir: Directory to save figures (if None, uses self.results_dir)
        """
        # Extract lesioned network data
        data_stim = results_stim['Lesioned Network']
        data_nostim = results_nostim['Lesioned Network']

        aligned_act = data_stim['aligned_activities']
        aligned_act_c = data_nostim['aligned_activities']
        aligned_hid = data_stim['aligned_hidden']
        aligned_hid_c = data_nostim['aligned_hidden']
        tids = data_stim['tids']

        plot_start_time = start_time + 1.2
        unique_targets = np.unique(tids)

        if save_dir is None:
            save_dir = self.results_dir

        # Build title suffix
        if neuron_description is not None:
            title_suffix = f"{neuron_description} Neuron"
        else:
            title_suffix = f"Neuron {neuron_idx}"

        # Plot 1: Mean activity and hidden by target (2x4 grid for 8 targets)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for target_idx, target_id in enumerate(sorted(unique_targets)):
            mask = tids == target_id
            mean_act = np.mean(aligned_act[mask], axis=0)[:, neuron_idx]
            mean_act_c = np.mean(aligned_act_c[mask], axis=0)[:, neuron_idx]
            mean_hid = np.mean(aligned_hid[mask], axis=0)[:, neuron_idx]
            mean_hid_c = np.mean(aligned_hid_c[mask], axis=0)[:, neuron_idx]

            if target_idx < 4:
                ax = axes[0, target_idx]
            else:
                ax = axes[1, target_idx - 4]

            self._plot_trial_on_axis_single(ax, mean_act, mean_act_c, mean_hid, mean_hid_c,
                                            plot_start_time, f"Target {target_id + 1}", self.dt)

        plt.suptitle(f'{title_suffix} - Single Neuron Response\nAmp={amplitude}, t={start_time}s', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            save_dir / f'single_neuron_{neuron_description.replace(" ", "_") if neuron_description else f"neuron{neuron_idx}"}_amp{amplitude}_t{start_time}.svg')
        plt.close()
        '''
        # Plot 2: Individual trials for this neuron (first 9 trials)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for trial_idx in range(min(9, aligned_act.shape[0])):
            ax = axes[trial_idx]
            trial_act = aligned_act[trial_idx, :, neuron_idx]
            trial_act_c = aligned_act_c[trial_idx, :, neuron_idx]
            trial_hid = aligned_hid[trial_idx, :, neuron_idx]
            trial_hid_c = aligned_hid_c[trial_idx, :, neuron_idx]
            self._plot_trial_on_axis_single(ax, trial_act, trial_act_c, trial_hid, trial_hid_c,
                                            plot_start_time, f"Trial {trial_idx}, Target {tids[trial_idx]}", self.dt)

        # Hide unused subplots
        for idx in range(aligned_act.shape[0], len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Single Neuron Individual Trials - Neuron {neuron_idx}\nAmp={amplitude}, t={start_time}s',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'single_neuron_trials_neuron{neuron_idx}_amp{amplitude}_t{start_time}.svg')
        plt.close()
        '''

    def _plot_trial_on_axis_single(self, ax, activity, activity_c, hidden, hidden_c, stim_time, title, dt):
        """Helper to plot single neuron response on given axis."""
        times = np.arange(len(activity)) * dt
        ax.plot(times, activity, 'r-', lw=1.5, label='with stim (activity)')
        ax.plot(times, activity_c, 'b-', lw=1.5, label='without stim (activity)')
        ax.plot(times, hidden, 'r--', lw=1.5, label='with stim (hidden)')
        ax.plot(times, hidden_c, 'b--', lw=1.5, label='without stim (hidden)')
        ax.axvspan(stim_time, stim_time + 0.2, alpha=0.3, color='cyan')  # duration = 200ms
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neural Activity / Activation State')
        ax.set_title(title)
        # ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def run_full_analysis(self, stim_config, save_plots=True, save_results=False, fixed_perm=None):
        """
        Run complete analysis pipeline.

        Args:
            stim_config: Stimulation configuration
            save_plots: If True, save plots to disk
            save_results: If True, save results to disk
            fixed_perm: Pre-generated lesion permutation (if None, generates random)

        """
        print("=" * 60)
        print("RUNNING BEHAVIORAL EXPERIMENTS ON ALL TEST TRIALS")
        print("=" * 60)

        # Run all conditions (returns results with stim and results_c without stim)
        results_stim, results_nostim = self.run_all_conditions(stim_config=stim_config, fixed_perm=fixed_perm)

        # Combine results for trajectory plotting (includes both healthy and lesioned)
        combined_results = {}
        order = ['Healthy Network_NoStim', 'Healthy Network_Stim', 'Lesioned Network_NoStim',
                 'Lesioned Network_Stim']
        for cond_name in results_nostim.keys():
            combined_results[f"{cond_name}_NoStim"] = results_nostim[cond_name]
        for cond_name in results_stim.keys():
            combined_results[f"{cond_name}_Stim"] = results_stim[cond_name]
        combined_results_ordered = {k: combined_results[k] for k in order if k in combined_results}

        if save_plots:
            # Plot trajectory comparison
            print("\n" + "=" * 60)
            print("PLOTTING TRAJECTORY ANALYSIS")
            # self.plot_trajectory_comparison(combined_results_ordered, individual_trials=False,plot_mode='points')
            self.plot_trajectory_comparison(combined_results_ordered, individual_trials=True,plot_mode='line')

            # Plot stimulation characterization (only if stimulation was applied)
            if stim_config.get('amplitude', 0) != 0:
                print("\n" + "=" * 60)
                print("PLOTTING STIMULATION CHARACTERIZATION")
                self.plot_stimulation_characterization(
                    results_stim, results_nostim,
                    amplitude=stim_config['amplitude'],
                    start_time=stim_config['start_time'],
                    save_path=self.results_dir / f'stim_heatmaps_amp{stim_config["amplitude"]}_t{stim_config["start_time"]}.svg'
                )

                # Plot distance response
                self.plot_distance_response(
                    results_stim, results_nostim,
                    amplitude=stim_config['amplitude'],
                    start_time=stim_config['start_time'],
                    save_path=self.results_dir / f'distance_response_amp{stim_config["amplitude"]}_t{stim_config["start_time"]}.svg'
                )

                # Get sorted neurons by distance
                sorted_indices, sorted_distances = self.get_sorted_neurons_by_distance()

                # Get 3 closest and 3 farthest neurons with descriptions
                closest_neurons = [
                    (sorted_indices[0], "Closest"),
                    (sorted_indices[1], "2nd Closest"),
                    (sorted_indices[2], "3rd Closest")
                ]
                farthest_neurons = [
                    (sorted_indices[-1], "Farthest"),
                    (sorted_indices[-2], "2nd Farthest"),
                    (sorted_indices[-3], "3rd Farthest")
                ]

                # Plot multi-target responses for each of the 6 neurons
                for neuron_idx, description in closest_neurons + farthest_neurons:
                    self.plot_multi_target_responses(
                        results_stim, results_nostim,
                        amplitude=stim_config['amplitude'],
                        start_time=stim_config['start_time'],
                        neuron_idx=neuron_idx,
                        neuron_description=description,
                        save_path=self.results_dir / f'multi_target_{description.replace(" ", "_")}_amp{stim_config["amplitude"]}_t{stim_config["start_time"]}.svg'
                    )

                # Explore single neuron response for each of the 6 neurons
                for neuron_idx, description in closest_neurons + farthest_neurons:
                    self.explore_single_neuron_response(
                        results_stim, results_nostim,
                        amplitude=stim_config['amplitude'],
                        start_time=stim_config['start_time'],
                        neuron_idx=neuron_idx,
                        neuron_description=description,
                        save_dir=self.results_dir
                    )

        # Compute jPCA for both conditions separately
        print("\n" + "=" * 60)
        print("COMPUTING jPCA")

        jpca_stim = {}
        jpca_nostim = {}

        time_window = (0, 2.5) if not self.fixed_movement_onset else (0, 3)

        for cond_name in results_nostim.keys():
            data = results_nostim[cond_name]
            if self.fixed_movement_onset:
                projected, total_var, plane_variances, rot_features, _, jpcs_var = compute_jpca_trajectories(
                    data['activities'],
                    n_jpcs=10,
                    time_window=time_window,
                    dt=self.dt
                )
            else:
                projected, total_var, plane_variances, rot_features, _, jpcs_var = compute_jpca_trajectories(
                    data['aligned_activities'],
                    n_jpcs=10,
                    time_window=time_window,
                    dt=self.dt
                )
            jpca_nostim[cond_name] = {
                'projected': projected,
                'tids': data['tids'],
                'total_variance_explained': total_var,
                'plane_variances': plane_variances,
                'rotational_features': rot_features,
                'jpcs_var': jpcs_var
            }

        for cond_name in results_stim.keys():
            data = results_stim[cond_name]
            if self.fixed_movement_onset:
                projected, total_var, plane_variances, rot_features, _, jpcs_var = compute_jpca_trajectories(
                    data['activities'],
                    n_jpcs=10,
                    time_window=time_window,
                    dt=self.dt
                )
            else:
                projected, total_var, plane_variances, rot_features, _, jpcs_var = compute_jpca_trajectories(
                    data['aligned_activities'],
                    n_jpcs=10,
                    time_window=time_window,
                    dt=self.dt
                )
            jpca_stim[cond_name] = {
                'projected': projected,
                'tids': data['tids'],
                'total_variance_explained': total_var,
                'plane_variances': plane_variances,
                'rotational_features': rot_features,
                'jpcs_var': jpcs_var
            }

        jpca_combined = {}
        order = ['Healthy Network_NoStim', 'Healthy Network_Stim', 'Lesioned Network_NoStim', 'Lesioned Network_Stim']
        for cond_name in results_nostim.keys():
            jpca_combined[f"{cond_name}_NoStim"] = jpca_nostim[cond_name]
        for cond_name in results_stim.keys():
            jpca_combined[f"{cond_name}_Stim"] = jpca_stim[cond_name]
        jpca_combined_ordered = {k: jpca_combined[k] for k in order if k in jpca_combined}

        if save_plots:
            # Plot jPCA across all planes
            # self.plot_jpca_comparison(jpca_combined_ordered, individual_trials=False,plot_mode='points')
            self.plot_jpca_comparison(jpca_combined_ordered, individual_trials=True, plot_mode='line')
        if save_results:
            # Save results
            savname = self.results_dir / 'behavior_activity_results'
            np.save(savname, {'stim': results_stim, 'nostim': results_nostim})

            savname1 = self.results_dir / 'jpca_results'
            np.save(savname1, jpca_combined)

        return results_stim, results_nostim, jpca_stim, jpca_nostim


def main():

    amp = 9

    base_stim_config = {
        'ID': '0',
        'sites': [(0.5, 0.5)],
        'transform': 'reciprocal',
        'combine': 'lin',
        'transform_params': {'max_distance': 0.03, 'power': 1},
        'reference_time': 'MovementOnset',
        'amplitude': amp,
        'stim_type': 'pulse',
        'start_time': 0,
        'duration': 0.2
    }

    MODEL_NAME = 'test_0'

    comparator = NetworkComparator(
        model_name=MODEL_NAME,
        results_dir='Plots' + str(amp),
        fixed_movement_onset=False
    )

    permutations, n_neurons, n_loaded = load_lesion_permutations(model_name=MODEL_NAME)
    comparator.run_full_analysis(base_stim_config,fixed_perm=permutations[5])

if __name__ == "__main__":
    main()