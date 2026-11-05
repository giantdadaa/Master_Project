# run_systematic_experiments.py

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from compare_networks import NetworkComparator
from generate_lesion_permutations import load_lesion_permutations


def main(name):
    # Define parameters
    amplitudes = range(-10,11)
    timings = [-1.0, -0.2, 0, 0.2]  # seconds relative to movement onset
    n_runs = 10
    n_nearest = 3
    n_farthest = 3

    # =========================================================
    # STEP 1: Load pre-generated fixed lesion permutations
    # =========================================================
    print("=" * 60)
    print("STEP 1: Loading fixed lesion permutations")
    print("=" * 60)

    # Load permutations (adjust path if needed)
    permutations, n_neurons, n_loaded = load_lesion_permutations(model_name=name)

    # Verify we have enough permutations
    if n_loaded < n_runs:
        print(f"Warning: Only {n_loaded} permutations available, but {n_runs} runs requested.")
        print(f"Using {n_loaded} runs instead.")
        n_runs = n_loaded

    # =========================================================
    # STEP 2: Run experiments with fixed lesion permutations
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 2: Running systematic experiments with fixed lesions")
    print("=" * 60)

    # Results storage
    all_results = []

    for amp in amplitudes:
        for timing in timings:
            for run_id in range(n_runs):
                print(f"\n{'=' * 60}")
                print(f"Running: amp={amp}, time={timing}, run={run_id}")
                print(f"{'=' * 60}")

                # Create unique results directory
                results_dir = f"systematic_amp{amp}_time{timing}_run{run_id}"

                # Stimulation config
                stim_config = {
                    'ID': '0',
                    'sites': [(0.5, 0.5)],
                    'transform': 'reciprocal',
                    'combine': 'lin',
                    'transform_params': {'max_distance': 0.03, 'power': 1},
                    'reference_time': 'MovementOnset',
                    'amplitude': amp,
                    'stim_type': 'pulse',
                    'start_time': timing,
                    'duration': 0.2
                }


                comparator = NetworkComparator(
                    model_name=name,
                    results_dir=results_dir,
                    fixed_movement_onset=False
                )

                # =========================================================
                # Apply FIXED lesion permutation for this run
                # =========================================================
                perm = permutations[run_id]

                # Run experiment
                if run_id == 5:
                    results_stim, results_nostim, jpca_stim, jpca_nostim = comparator.run_full_analysis(
                        stim_config,
                        save_plots=True,
                        fixed_perm=perm
                    )
                else:
                    results_stim, results_nostim, jpca_stim, jpca_nostim = comparator.run_full_analysis(
                        stim_config,
                        save_plots=False,
                        fixed_perm=perm
                    )

                # =========================================================
                # Extract metrics
                # =========================================================

                lesioned_stim = results_stim['Lesioned Network']
                lesioned_nostim = results_nostim['Lesioned Network']
                healthy_stim = results_stim['Healthy Network']
                healthy_nostim = results_nostim['Healthy Network']

                jpca_lesioned_stim = jpca_stim['Lesioned Network']
                jpca_lesioned_nostim = jpca_nostim['Lesioned Network']
                jpca_healthy_stim = jpca_stim['Healthy Network']
                jpca_healthy_nostim = jpca_nostim['Healthy Network']

                plane_vars_lesioned_stim = jpca_lesioned_stim['plane_variances']
                plane_vars_lesioned_nostim = jpca_lesioned_nostim['plane_variances']
                plane_vars_healthy_stim = jpca_healthy_stim['plane_variances']
                plane_vars_healthy_nostim = jpca_healthy_nostim['plane_variances']

                rot_feat_lesioned_stim = jpca_lesioned_stim['rotational_features']
                rot_feat_lesioned_nostim = jpca_lesioned_nostim['rotational_features']
                rot_feat_healthy_stim = jpca_healthy_stim['rotational_features']
                rot_feat_healthy_nostim = jpca_healthy_nostim['rotational_features']

                # Get sorted neurons by distance
                sorted_indices, sorted_distances = comparator.get_sorted_neurons_by_distance()
                nearest_indices = sorted_indices[:n_nearest]
                farthest_indices = sorted_indices[-n_farthest:]
                nearest_distances = sorted_distances[:n_nearest]
                farthest_distances = sorted_distances[-n_farthest:]

                # Compute peak responses
                plot_start_time = timing + 1.2
                activity_peaks, hidden_peaks = comparator._compute_peak_responses(
                    lesioned_stim['aligned_activities'], lesioned_nostim['aligned_activities'],
                    lesioned_stim['aligned_hidden'], lesioned_nostim['aligned_hidden'],
                    plot_start_time, amp
                )

                # =========================================================
                # Build result entry
                # =========================================================

                result_entry = {
                    'amplitude': amp,
                    'timing': timing,
                    'run_id': run_id,

                    # Behavioral metrics
                    'lesioned_endpoint_stim': lesioned_stim['trajectory_metrics']['endpoint_error_mean'],
                    'lesioned_straightness_stim': lesioned_stim['trajectory_metrics']['straightness_mean'],
                    'lesioned_maxdev_stim': lesioned_stim['trajectory_metrics']['max_deviation_mean'],

                    'lesioned_endpoint_nostim': lesioned_nostim['trajectory_metrics']['endpoint_error_mean'],
                    'lesioned_straightness_nostim': lesioned_nostim['trajectory_metrics']['straightness_mean'],
                    'lesioned_maxdev_nostim': lesioned_nostim['trajectory_metrics']['max_deviation_mean'],

                    'healthy_endpoint_stim': healthy_stim['trajectory_metrics']['endpoint_error_mean'],
                    'healthy_straightness_stim': healthy_stim['trajectory_metrics']['straightness_mean'],
                    'healthy_maxdev_stim': healthy_stim['trajectory_metrics']['max_deviation_mean'],

                    'healthy_endpoint_nostim': healthy_nostim['trajectory_metrics']['endpoint_error_mean'],
                    'healthy_straightness_nostim': healthy_nostim['trajectory_metrics']['straightness_mean'],
                    'healthy_maxdev_nostim': healthy_nostim['trajectory_metrics']['max_deviation_mean'],

                    # jPCA total variance
                    'jpca_total_var_lesioned_stim': jpca_lesioned_stim['total_variance_explained'],
                    'jpca_total_var_lesioned_nostim': jpca_lesioned_nostim['total_variance_explained'],
                    'jpca_total_var_healthy_stim': jpca_healthy_stim['total_variance_explained'],
                    'jpca_total_var_healthy_nostim': jpca_healthy_nostim['total_variance_explained'],

                    # jPCA plane variances
                    'jpca_plane1_var_lesioned_stim': plane_vars_lesioned_stim[0] if len(plane_vars_lesioned_stim) > 0 else np.nan,
                    'jpca_plane2_var_lesioned_stim': plane_vars_lesioned_stim[1] if len(plane_vars_lesioned_stim) > 1 else np.nan,
                    'jpca_plane3_var_lesioned_stim': plane_vars_lesioned_stim[2] if len(plane_vars_lesioned_stim) > 2 else np.nan,
                    'jpca_plane4_var_lesioned_stim': plane_vars_lesioned_stim[3] if len(plane_vars_lesioned_stim) > 3 else np.nan,
                    'jpca_plane5_var_lesioned_stim': plane_vars_lesioned_stim[4] if len(plane_vars_lesioned_stim) > 4 else np.nan,

                    'jpca_plane1_var_lesioned_nostim': plane_vars_lesioned_nostim[0] if len(plane_vars_lesioned_nostim) > 0 else np.nan,
                    'jpca_plane2_var_lesioned_nostim': plane_vars_lesioned_nostim[1] if len(plane_vars_lesioned_nostim) > 1 else np.nan,
                    'jpca_plane3_var_lesioned_nostim': plane_vars_lesioned_nostim[2] if len(plane_vars_lesioned_nostim) > 2 else np.nan,
                    'jpca_plane4_var_lesioned_nostim': plane_vars_lesioned_nostim[3] if len(plane_vars_lesioned_nostim) > 3 else np.nan,
                    'jpca_plane5_var_lesioned_nostim': plane_vars_lesioned_nostim[4] if len(plane_vars_lesioned_nostim) > 4 else np.nan,

                    'jpca_plane1_var_healthy_stim': plane_vars_healthy_stim[0] if len(plane_vars_healthy_stim) > 0 else np.nan,
                    'jpca_plane2_var_healthy_stim': plane_vars_healthy_stim[1] if len(plane_vars_healthy_stim) > 1 else np.nan,
                    'jpca_plane3_var_healthy_stim': plane_vars_healthy_stim[2] if len(plane_vars_healthy_stim) > 2 else np.nan,
                    'jpca_plane4_var_healthy_stim': plane_vars_healthy_stim[3] if len(plane_vars_healthy_stim) > 3 else np.nan,
                    'jpca_plane5_var_healthy_stim': plane_vars_healthy_stim[4] if len(plane_vars_healthy_stim) > 4 else np.nan,

                    'jpca_plane1_var_healthy_nostim': plane_vars_healthy_nostim[0] if len(plane_vars_healthy_nostim) > 0 else np.nan,
                    'jpca_plane2_var_healthy_nostim': plane_vars_healthy_nostim[1] if len(plane_vars_healthy_nostim) > 1 else np.nan,
                    'jpca_plane3_var_healthy_nostim': plane_vars_healthy_nostim[2] if len(plane_vars_healthy_nostim) > 2 else np.nan,
                    'jpca_plane4_var_healthy_nostim': plane_vars_healthy_nostim[3] if len(plane_vars_healthy_nostim) > 3 else np.nan,
                    'jpca_plane5_var_healthy_nostim': plane_vars_healthy_nostim[4] if len(plane_vars_healthy_nostim) > 4 else np.nan,

                    # jPCA angular velocities
                    'jpca_angvel_plane1_lesioned_stim': rot_feat_lesioned_stim['plane_1']['mean_angular_velocity'],
                    'jpca_angvel_plane2_lesioned_stim': rot_feat_lesioned_stim.get('plane_2', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane3_lesioned_stim': rot_feat_lesioned_stim.get('plane_3', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane4_lesioned_stim': rot_feat_lesioned_stim.get('plane_4', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane5_lesioned_stim': rot_feat_lesioned_stim.get('plane_5', {}).get('mean_angular_velocity', np.nan),

                    'jpca_angvel_plane1_lesioned_nostim': rot_feat_lesioned_nostim['plane_1']['mean_angular_velocity'],
                    'jpca_angvel_plane2_lesioned_nostim': rot_feat_lesioned_nostim.get('plane_2', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane3_lesioned_nostim': rot_feat_lesioned_nostim.get('plane_3', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane4_lesioned_nostim': rot_feat_lesioned_nostim.get('plane_4', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane5_lesioned_nostim': rot_feat_lesioned_nostim.get('plane_5', {}).get('mean_angular_velocity', np.nan),

                    'jpca_angvel_plane1_healthy_stim': rot_feat_healthy_stim['plane_1']['mean_angular_velocity'],
                    'jpca_angvel_plane2_healthy_stim': rot_feat_healthy_stim.get('plane_2', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane3_healthy_stim': rot_feat_healthy_stim.get('plane_3', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane4_healthy_stim': rot_feat_healthy_stim.get('plane_4', {}).get('mean_angular_velocity', np.nan),
                    'jpca_angvel_plane5_healthy_stim': rot_feat_healthy_stim.get('plane_5', {}).get('mean_angular_velocity', np.nan),

                    'jpca_angvel_plane1_healthy_nostim': rot_feat_healthy_nostim['plane_1']['mean_angular_velocity'],
                    'jpca_angvel_plane2_healthy_nostim': rot_feat_healthy_nostim.get('plane_2', {}).get('mean_angular_velocity',np.nan),
                    'jpca_angvel_plane3_healthy_nostim': rot_feat_healthy_nostim.get('plane_3', {}).get('mean_angular_velocity',np.nan),
                    'jpca_angvel_plane4_healthy_nostim': rot_feat_healthy_nostim.get('plane_4', {}).get('mean_angular_velocity',np.nan),
                    'jpca_angvel_plane5_healthy_nostim': rot_feat_healthy_nostim.get('plane_5', {}).get('mean_angular_velocity',np.nan),

                    # jPCA rotational frequencies
                    'jpca_freq_plane1_lesioned_stim': rot_feat_lesioned_stim['plane_1']['mean_rotational_frequency'],
                    'jpca_freq_plane2_lesioned_stim': rot_feat_lesioned_stim.get('plane_2', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane3_lesioned_stim': rot_feat_lesioned_stim.get('plane_3', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane4_lesioned_stim': rot_feat_lesioned_stim.get('plane_4', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane5_lesioned_stim': rot_feat_lesioned_stim.get('plane_5', {}).get('mean_rotational_frequency', np.nan),

                    'jpca_freq_plane1_lesioned_nostim': rot_feat_lesioned_nostim['plane_1']['mean_rotational_frequency'],
                    'jpca_freq_plane2_lesioned_nostim': rot_feat_lesioned_nostim.get('plane_2', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane3_lesioned_nostim': rot_feat_lesioned_nostim.get('plane_3', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane4_lesioned_nostim': rot_feat_lesioned_nostim.get('plane_4', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane5_lesioned_nostim': rot_feat_lesioned_nostim.get('plane_5', {}).get('mean_rotational_frequency', np.nan),

                    'jpca_freq_plane1_healthy_stim': rot_feat_healthy_stim['plane_1']['mean_rotational_frequency'],
                    'jpca_freq_plane2_healthy_stim': rot_feat_healthy_stim.get('plane_2', {}).get('mean_rotational_frequency',np.nan),
                    'jpca_freq_plane3_healthy_stim': rot_feat_healthy_stim.get('plane_3', {}).get('mean_rotational_frequency',np.nan),
                    'jpca_freq_plane4_healthy_stim': rot_feat_healthy_stim.get('plane_4', {}).get('mean_rotational_frequency',np.nan),
                    'jpca_freq_plane5_healthy_stim': rot_feat_healthy_stim.get('plane_5', {}).get('mean_rotational_frequency',np.nan),

                    'jpca_freq_plane1_healthy_nostim': rot_feat_healthy_nostim['plane_1']['mean_rotational_frequency'],
                    'jpca_freq_plane2_healthy_nostim': rot_feat_healthy_nostim.get('plane_2', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane3_healthy_nostim': rot_feat_healthy_nostim.get('plane_3', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane4_healthy_nostim': rot_feat_healthy_nostim.get('plane_4', {}).get('mean_rotational_frequency', np.nan),
                    'jpca_freq_plane5_healthy_nostim': rot_feat_healthy_nostim.get('plane_5', {}).get('mean_rotational_frequency', np.nan),
                }

                # Add per-neuron data
                for i, (idx, dist) in enumerate(zip(nearest_indices, nearest_distances)):
                    result_entry[f'nearest_{i}_idx'] = int(idx)
                    result_entry[f'nearest_{i}_dist'] = dist
                    result_entry[f'nearest_{i}_activity_peak'] = activity_peaks[idx]
                    result_entry[f'nearest_{i}_hidden_peak'] = hidden_peaks[idx]

                for i, (idx, dist) in enumerate(zip(farthest_indices, farthest_distances)):
                    result_entry[f'farthest_{i}_idx'] = int(idx)
                    result_entry[f'farthest_{i}_dist'] = dist
                    result_entry[f'farthest_{i}_activity_peak'] = activity_peaks[idx]
                    result_entry[f'farthest_{i}_hidden_peak'] = hidden_peaks[idx]

                all_results.append(result_entry)

    # Save to DataFrame
    from pathlib import Path

    # Create save directory
    save_dir = Path.cwd() / 'results' / name / 'systematic_results'
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save CSV
    df=pd.DataFrame(all_results)
    csv_path = save_dir / 'systematic_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {csv_path}")
    print(f"Total experiments: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main(name='test_0')