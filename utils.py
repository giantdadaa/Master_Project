#!/usr/bin/env python3

import os
import numpy as np
import torch
from pathlib import Path
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.interpolate import interp1d
import pickle

from modeldef import RNN
from jPCA import JPCA


def load_trained_model(name, phase='phase3_training'):
    """Load pre-trained model from saved checkpoint."""
    directory = Path.cwd() / 'results' / name
    dataT = torch.load(directory / phase, map_location='cpu', weights_only=False)
    params = np.load(directory / 'params.npy', allow_pickle=True).item()

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    model = RNN(
        params['model']['input_dim'],
        params['model']['output_dim'],
        params['model']['n'],
        params['model']['dt'] / params['model']['tau'],
        dtype,
        params['model']['dt'],
        params['model']['fwd_delay'],
        params['model']['fb_delay'],
        fb_sparsity=params['model']['fb_sparsity'],
        nonlin=params['model']['nonlin'],
        noiseout=params['model']['noise_amp'],
        noise_kernel_size=params['model']['noise_kernel_size'],
        noisein=params['model']['noise_stim_amp'],
        rec_sparsity=params['model']['rec_sparsity'])

    model.load_state_dict(dataT['model_state_dict'])
    model.eval()
    model.biolearning = False

    return model, params


def create_stimulation_intensity(tsteps, n_trials, movement_onset, stim_type='pulse',
                                 amplitude=0.1, reference_time='MovementOnset', start_time=0.2, duration=0.2, dt=0.01,
                                 n_repetition=1, IPI=0.2):

    stim_intensity = torch.zeros(tsteps, n_trials)

    if stim_type == 'none' or amplitude == 0:
        return stim_intensity

    if stim_type == 'constant':
        stim_intensity[:, :] = amplitude
        return stim_intensity

    if reference_time == 'MovementOnset':
        start_idx = np.zeros((n_trials, n_repetition), dtype=int)
        end_idx = np.zeros((n_trials, n_repetition), dtype=int)

        for trial_idx, onset_idx in enumerate(movement_onset):

            # Calculate start indices for each repetition
            start_times = np.linspace(
                onset_idx + start_time /dt,
                onset_idx + (start_time + (n_repetition - 1) * (duration + IPI)) / dt,
                n_repetition
            )
            start_idx[trial_idx] = start_times.astype(int)
            end_idx[trial_idx] = (start_idx[trial_idx] + duration / dt).astype(int)

            for n_rep in range(n_repetition):
                s_idx = int(start_idx[trial_idx, n_rep])
                e_idx = int(end_idx[trial_idx, n_rep])

                if s_idx >= tsteps:
                    continue
                e_idx = min(e_idx, tsteps)

                if stim_type == 'pulse':
                    stim_intensity[s_idx:e_idx, trial_idx] = amplitude
                elif stim_type == 'ramp':
                    ramp = torch.linspace(0, amplitude, e_idx - s_idx)
                    stim_intensity[s_idx:e_idx, trial_idx] = ramp.unsqueeze(1)

    elif reference_time == 'TrialOnset':
        start_idx = np.linspace(int(start_time / dt),
                                int((start_time + (n_repetition - 1) * (duration + IPI)) / dt),
                                n_repetition).astype(int)
        end_idx = (start_idx + int(duration / dt)).astype(int)

        for n_rep in range(n_repetition):
            s_idx = int(start_idx[n_rep])
            e_idx = int(end_idx[n_rep])

            if s_idx >= tsteps:
                continue
            e_idx = min(e_idx, tsteps)

            if stim_type == 'pulse':
                stim_intensity[s_idx:e_idx, :] = amplitude
            elif stim_type == 'ramp':
                ramp = torch.linspace(0, amplitude, e_idx - s_idx)
                stim_intensity[s_idx:e_idx, :] = ramp.unsqueeze(1)

    return stim_intensity


def setup_stimulation(model, stimulation_config, tsteps, dt, movement_onset):
    """Set up stimulation weights and intensity."""

    # Generate random neuron positions if not already present
    if model.neuron_positions is None:
        model.generate_random_neuron_positions()

    # Get stimulation sites
    if 'sites' in stimulation_config:
        sites = stimulation_config['sites']
    else:
        sites = [(0.5, 0.5)]

    transform = stimulation_config.get('transform', 'reciprocal')
    combine = stimulation_config.get('combine', 'sum')
    transform_params = stimulation_config.get('transform_params', {})

    # Set stimulation weights
    stim_weights = model.set_stim_weights_from_sites(
        sites, transform=transform, combine=combine, **transform_params
    )

    # Create stimulation intensity
    stim_intensity = create_stimulation_intensity(
        tsteps=tsteps,
        n_trials=len(movement_onset),
        movement_onset=movement_onset,
        dt=dt,
        stim_type=stimulation_config.get('stim_type', 'pulse'),
        amplitude=stimulation_config.get('amplitude', 0.2),
        reference_time=stimulation_config.get('reference_time', 'MovementOnset'),
        start_time=stimulation_config.get('start_time', 0.2),
        duration=stimulation_config.get('duration', 0.2)
    )

    return stim_weights, stim_intensity


def compute_position_from_velocity(velocities, dt=0.01):
    """
    Integrate velocity to get position.

    Args:
        velocities: [n_trials, time, 2] or [time, 2]
        dt: time step

    Returns:
        positions: same shape as velocities
    """
    if velocities.ndim == 2:
        velocities = velocities[np.newaxis, :, :]
        single_trial = True
    else:
        single_trial = False

    n_trials, n_time, _ = velocities.shape
    positions = np.zeros_like(velocities)

    for trial in range(n_trials):
        pos = np.zeros((n_time, 2))
        for t in range(1, n_time):
            pos[t] = pos[t - 1] + velocities[trial, t] * dt
        positions[trial] = pos

    return positions[0] if single_trial else positions


def compute_trajectory_metrics(positions, tids, params):
    """
    Compute detailed trajectory metrics for all targets simultaneously.

    Args:
        positions: [n_trials, time, 2] - positions for all trials
        tids: [n_trials] - target IDs for each trial
        ntargets: Number of targets (default 8)
        output_range: Distance to targets (default 5 cm)

    Returns:
        dict with various trajectory metrics for all trials
    """

    ntargets = params['data']['center-out-reach']['ntargets']
    output_range = params['data']['center-out-reach']['output_range']
    dt = params['model']['dt']
    n_trials, n_time, _ = positions.shape

    # Precompute target positions for all targets
    phi = np.linspace(0, 2 * np.pi, ntargets, endpoint=False)
    target_positions = output_range * np.array([np.cos(phi), np.sin(phi)]).T  # [ntargets, 2]

    # Initialize metrics arrays
    endpoint_errors = np.zeros(n_trials)
    path_lengths = np.zeros(n_trials)
    straightness_ratios = np.zeros(n_trials)
    max_deviations = np.zeros(n_trials)
    mean_trajectories = np.zeros((ntargets, n_time, 2))
    trajectory_variabilities = np.zeros((ntargets, n_time, 2))

    # Process each target separately for mean trajectories
    for target_id in range(ntargets):
        target_mask = tids == target_id
        if np.sum(target_mask) > 0:
            target_positions_subset = positions[target_mask]
            mean_trajectories[target_id] = np.mean(target_positions_subset, axis=0)
            trajectory_variabilities[target_id] = np.std(target_positions_subset, axis=0)

    # Compute metrics for each trial
    for trial_idx in range(n_trials):
        pos = positions[trial_idx]
        target_id = tids[trial_idx]
        target_position = target_positions[target_id]

        # 1. Endpoint error (final position accuracy)
        endpoint_errors[trial_idx] = np.linalg.norm(pos[-1] - target_position)

        # 2. Path length (total distance traveled)
        diffs = np.diff(pos, axis=0)
        path_lengths[trial_idx] = np.sum(np.linalg.norm(diffs, axis=1))

        # 3. Straightness ratio (path length / straight-line distance)
        straight_distance = np.linalg.norm(target_position - pos[0])
        straightness_ratios[trial_idx] = path_lengths[
                                             trial_idx] / straight_distance if straight_distance > 0 else np.nan

        # 4. Maximum deviation from ideal straight line
        start = pos[0]
        direction = target_position - start
        direction_norm = direction / np.linalg.norm(direction)

        vectors = pos - start
        projections = np.dot(vectors, direction_norm)
        closest_points = start + np.outer(projections, direction_norm)
        deviations = np.linalg.norm(pos - closest_points, axis=1)
        max_deviations[trial_idx] = np.max(deviations)

    # Compute velocities from positions for all trials
    velocities = np.zeros_like(positions)
    velocities[:, 1:, :] = np.diff(positions, axis=1) / dt

    # Overall metrics across all trials
    results = {
        'endpoint_errors': endpoint_errors,
        'endpoint_error_mean': np.mean(endpoint_errors),
        'endpoint_error_std': np.std(endpoint_errors),
        'path_lengths': path_lengths,
        'path_length_mean': np.mean(path_lengths),
        'path_length_std': np.std(path_lengths),
        'straightness_ratios': straightness_ratios,
        'straightness_mean': np.mean(straightness_ratios),
        'straightness_std': np.std(straightness_ratios),
        'max_deviations': max_deviations,
        'max_deviation_mean': np.mean(max_deviations),
        'max_deviation_std': np.std(max_deviations),
        'trajectories': positions,
        'velocities': velocities,
        'mean_trajectories': mean_trajectories,
        'trajectory_variabilities': trajectory_variabilities,
        'target_positions': target_positions
    }

    return results


def run_reaching_experiment(model, params, stimulation_config=None, compare=False,
                            fixed_movement_onset=False, ntrials=200, target_id=None):
    """
    Run reaching experiment with optional stimulation on ALL 8 targets or a single target.

    Args:
        model: RNN model
        params: Model parameters dictionary
        stimulation_config: Stimulation configuration (None for no stimulation)
        compare: If True, also run comparison without stimulation
        fixed_movement_onset: If True, use fixed movement onset dataset
        ntrials: Number of trials to run (sampled from test set)
        target_id: Target specification:
            - None or 'all': Use all 8 targets (default behavior)
            - int (0-7): Use only the specified single target
            - list of ints: Use only the specified targets

    Returns:
        results: Dictionary with neural activities, outputs, and metrics
        results_compare: Same as results but without stimulation (if compare=True)

    Notes:
        - When target_id is specified (e.g., target_id=0), only trials with that target
          are used. The number of trials will be min(ntrials, available_trials_for_target).
        - When target_id is None or 'all', trials are sampled uniformly from all targets.
    """
    tsteps = params['model']['tsteps']
    dt = params['model']['dt']
    ntargets = params['data']['center-out-reach']['ntargets']

    # Load data
    directory = Path.cwd() / 'results' / params['name']
    if fixed_movement_onset:
        data_path = directory / 'fixed_movement_onset_data.npy'
        data = np.load(data_path, allow_pickle=True).item()
    else:
        data_path = directory / 'data.npy'
        data = np.load(data_path, allow_pickle=True).item()['center-out-reach']
    testdata = data['test_set']

    # Get full data
    all_tids = testdata['tids']
    all_targets = testdata['target']
    all_stimuli = testdata['stimulus']
    all_peak_speed = testdata['peak_speed']

    # Filter by target if specified
    if target_id is not None and target_id != 'all':
        # Convert single int to list for uniform handling
        if isinstance(target_id, int):
            target_ids = [target_id]
        elif isinstance(target_id, list):
            target_ids = target_id
        else:
            raise ValueError(f"target_id must be None, 'all', int, or list of ints. Got {type(target_id)}")

        # Create mask for selected targets
        target_mask = np.isin(all_tids, target_ids)
        filtered_tids = all_tids[target_mask]
        filtered_targets = all_targets[target_mask]
        filtered_stimuli = all_stimuli[target_mask]
        filtered_peak_speed = all_peak_speed[target_mask]

        print(f"Filtered to targets {target_ids}: {len(filtered_tids)} trials available")

        # Sample trials (ntrials or all available)
        n_available = len(filtered_tids)
        n_to_use = min(ntrials, n_available)
        trial_idx = np.random.choice(n_available, n_to_use, replace=False)

        tout = filtered_targets[trial_idx, :, 2:]
        tstim = filtered_stimuli[trial_idx]
        tids = filtered_tids[trial_idx]
        go_on = filtered_peak_speed[trial_idx]
    else:
        # Use all targets (original behavior)
        n_available = len(all_tids)
        n_to_use = min(ntrials, n_available)
        trial_idx = np.random.choice(n_available, n_to_use, replace=False)

        tout = all_targets[trial_idx, :, 2:]
        tstim = all_stimuli[trial_idx]
        tids = all_tids[trial_idx]
        go_on = all_peak_speed[trial_idx]

    # Calculate movement indices
    movement_onset = go_on - 50
    start_idx = go_on - 170
    end_idx = go_on + 80

    # Setup stimulation if configured
    if stimulation_config is not None and stimulation_config.get('amplitude', 0) != 0:
        stim_weights, stim_intensity = setup_stimulation(model, stimulation_config, tsteps, dt, movement_onset)

        # Plot stimulation and neurons' sites
        save_dir = Path.cwd() / 'results' / params['name'] / ('stimulation_' + stimulation_config['ID'])
        save_dir.mkdir(exist_ok=True)
        model.visualize_stimulation_profile(save_path=save_dir / 'stimulation_profile.svg')
    else:
        stim_intensity = torch.zeros(tsteps, len(movement_onset))
        stim_weights = None

    # Transpose to [tsteps, n_trials, dim] as expected by model.forward()
    stim = torch.Tensor(tstim.transpose(1, 0, 2)).type(model.dtype)
    pert = torch.zeros(tout.transpose(1, 0, 2).shape).type(model.dtype)

    def run_model():
        # Run model on ALL test trials at once (batch processing)
        with torch.no_grad():
            testout, testl1, hidden = model(stim, pert, stim_intensity=stim_intensity)

        # Get velocity output
        velocity = model.get_output(testl1).cpu().detach().numpy()  # [tsteps, n_trials, 2]
        velocity = velocity.transpose(1, 0, 2)  # [n_trials, tsteps, 2]

        # Get activity
        activity = testl1.cpu().detach().numpy().transpose(1, 0, 2)  # [n_trials, tsteps, neurons]
        hidden_state = hidden.cpu().detach().numpy().transpose(1, 0, 2)

        aligned_activities = np.zeros((activity.shape[0], 250, params['model']['n']))
        aligned_hidden = np.zeros((hidden_state.shape[0], 250, params['model']['n']))
        for n_trial in range(activity.shape[0]):
            aligned_activities[n_trial] = activity[n_trial, start_idx[n_trial]:end_idx[n_trial], :]
            aligned_hidden[n_trial] = hidden_state[n_trial, start_idx[n_trial]:end_idx[n_trial], :]

        # Integrate velocity to get position for each trial
        positions = []
        for trial_idx in range(velocity.shape[0]):
            pos = compute_position_from_velocity(velocity[trial_idx], dt)
            positions.append(pos)
        positions = np.array(positions)  # [n_trials, tsteps, 2]

        # Compute trajectory metrics for ALL trials at once
        trajectory_metrics = compute_trajectory_metrics(positions, tids, params)

        # Aggregate results by target (using the precomputed mean trajectories)
        per_target_results = {}
        for target_id_val in range(ntargets):
            target_mask = tids == target_id_val
            if np.sum(target_mask) > 0:
                per_target_results[target_id_val] = {
                    'activities': activity[target_mask],
                    'aligned_activities': aligned_activities[target_mask] if fixed_movement_onset == False else None,
                    'hidden_state': hidden_state[target_mask],
                    'aligned_hidden': aligned_hidden[target_mask] if fixed_movement_onset == False else None,
                    'outputs': velocity[target_mask],
                    'positions': positions[target_mask],
                    'mean_trajectory': trajectory_metrics['mean_trajectories'][target_id_val],
                    'trajectory_variability': trajectory_metrics['trajectory_variabilities'][target_id_val],
                    'target_position': trajectory_metrics['target_positions'][target_id_val],
                    'target_angle': target_id_val * 45,
                    'trajectory_metrics': {
                        'endpoint_error_mean': np.mean(trajectory_metrics['endpoint_errors'][target_mask]),
                        'endpoint_error_std': np.std(trajectory_metrics['endpoint_errors'][target_mask]),
                        'straightness_mean': np.mean(trajectory_metrics['straightness_ratios'][target_mask]),
                        'straightness_std': np.std(trajectory_metrics['straightness_ratios'][target_mask]),
                        'max_deviation_mean': np.mean(trajectory_metrics['max_deviations'][target_mask]),
                        'max_deviation_std': np.std(trajectory_metrics['max_deviations'][target_mask]),
                    },
                }

        results = {
            'activities': activity,
            'aligned_activities': aligned_activities,
            'hidden_state': hidden_state,
            'aligned_hidden': aligned_hidden,
            'outputs': velocity,
            'positions': positions,
            'tids': tids,
            'trajectory_metrics': {
                'endpoint_error_mean': trajectory_metrics['endpoint_error_mean'],
                'endpoint_error_std': trajectory_metrics['endpoint_error_std'],
                'straightness_mean': trajectory_metrics['straightness_mean'],
                'straightness_std': trajectory_metrics['straightness_std'],
                'max_deviation_mean': trajectory_metrics['max_deviation_mean'],
                'max_deviation_std': trajectory_metrics['max_deviation_std'],
            },
            'stimulation_config': stimulation_config if stim_weights is not None else None,
            'stimulation_weights': stim_weights,
            'stimulation_intensity': stim_intensity,
            'per_target_results': per_target_results
        }
        return results

    results = run_model()
    results_compare = None
    if compare:
        stim_intensity = torch.zeros(tsteps, len(movement_onset))
        stim_weights = None
        results_compare = run_model()

    return results, results_compare


def compute_jpca_trajectories(activities, n_jpcs=10, time_window=(0, 2.5), dt=0.01):
    """
    Compute jPCA trajectories for neural activity.

    Args:
        activities: [n_trials, time, neurons]
        n_jpcs: Number of jPCs (must be even, default 10 for 5 planes)
        time_window: (start_time, end_time) in seconds
        dt: time step

    Returns:
        projected: List of [time, n_jpcs] arrays for each trial
        variance_explained: Variance explained by ALL jPCs
        plane_variances: Variance explained for each plane (5 values)
        rotational_features: Dict with features for each plane
        jpca_object: Fitted JPCA object
        jpcs_var: Variance explained by individual jPCs
    """
    # Extract time window
    start_idx = int(time_window[0] / dt)
    end_idx = int(time_window[1] / dt)
    activities_cropped = activities[:, start_idx:end_idx, :]

    # Convert to list of trials for jPCA
    datas = [activities_cropped[trial_idx] for trial_idx in range(activities_cropped.shape[0])]

    # Create and fit jPCA
    jpca = JPCA(num_jpcs=n_jpcs)
    times = list(np.arange(activities_cropped.shape[1]) * dt)

    projected, full_var, pca_var, jpca_var = jpca.fit(
        datas, times=times, tstart=times[0], tend=times[-1],
        pca=True, num_pcs=n_jpcs, subtract_cc_mean=True
    )

    # Total variance explained by all jPCs
    total_variance_explained = np.sum(jpca_var) / full_var

    # Calculate variance explained PER jPCs
    jpcs_var = jpca_var / full_var

    # Calculate variance explained PER PLANE (pairs of jPCs)
    n_planes = n_jpcs // 2
    plane_variances = []
    for plane_idx in range(n_planes):
        plane_var = np.sum(jpca_var[plane_idx * 2:(plane_idx + 1) * 2]) / full_var
        plane_variances.append(plane_var)

    # Extract rotational features for each plane
    rotational_features = {}

    for plane_idx in range(n_planes):
        # Get indices for this plane
        idx1 = plane_idx * 2
        idx2 = plane_idx * 2 + 1

        angles = []
        radii = []
        angular_velocities = []
        rotational_frequency = []

        for proj in projected:
            # Take the 2D trajectory for this plane
            traj_2d = proj[:, [idx1, idx2]]

            # Compute angle in this plane
            ang = np.arctan2(traj_2d[:, 1], traj_2d[:, 0])
            angles.append(ang)

            # Compute radius (distance from origin)
            rad = np.linalg.norm(traj_2d, axis=1)
            radii.append(rad)

            # Compute angular velocity
            ang_unwrapped = np.unwrap(ang)
            ang_vel = np.diff(ang_unwrapped) / dt
            angular_velocities.append(ang_vel)

            # Compute rotational frequency
            rotational_frequency.append(ang_vel / (2*np.pi))

        rotational_features[f'plane_{plane_idx + 1}'] = {
            'mean_angular_velocity': np.mean([np.mean(v) for v in angular_velocities]),
            'std_angular_velocity': np.std([np.mean(v) for v in angular_velocities]),
            'mean_rotational_frequency': np.mean([np.mean(f) for f in rotational_frequency]),
            'std_rotational_frequency': np.std([np.mean(f) for f in rotational_frequency]),
            'mean_radius': np.mean([np.mean(r) for r in radii]),
            'std_radius': np.std([np.mean(r) for r in radii]),
            'rotation_consistency': 1 - np.std([np.mean(v) for v in angular_velocities]) /
                                    abs(np.mean([np.mean(v) for v in angular_velocities]))
            if abs(np.mean([np.mean(v) for v in angular_velocities])) > 0 else 0
        }

    return projected, total_variance_explained, plane_variances, rotational_features, jpca, jpcs_var