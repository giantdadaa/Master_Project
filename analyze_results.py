
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_compute_metrics(csv_path):
    """Load results and compute derived metrics (differences)."""
    df = pd.read_csv(csv_path)

    # Compute differences for lesioned (stim - nostim) - negative means improvement
    df['lesioned_endpoint_diff'] = df['lesioned_endpoint_stim'] - df['lesioned_endpoint_nostim']
    df['lesioned_maxdev_diff'] = df['lesioned_maxdev_stim'] - df['lesioned_maxdev_nostim']
    df['lesioned_straightness_diff'] = df['lesioned_straightness_stim'] - df['lesioned_straightness_nostim']

    # Compute differences for healthy (stim - nostim)
    df['healthy_endpoint_diff'] = df['healthy_endpoint_stim'] - df['healthy_endpoint_nostim']
    df['healthy_maxdev_diff'] = df['healthy_maxdev_stim'] - df['healthy_maxdev_nostim']
    df['healthy_straightness_diff'] = df['healthy_straightness_stim'] - df['healthy_straightness_nostim']

    return df


def compute_summary_stats(df):
    """Compute mean, std, and CV across runs for each condition."""
    grouped = df.groupby(['amplitude', 'timing'])

    summary = []
    for (amp, timing), group in grouped:
        entry = {
            'amplitude': amp,
            'timing': timing,
            'n_runs': len(group),

            # ===== LESIONED BEHAVIORAL METRICS =====
            'lesioned_endpoint_mean': group['lesioned_endpoint_stim'].mean(),
            'lesioned_endpoint_std': group['lesioned_endpoint_stim'].std(),

            'lesioned_nostim_endpoint_mean': group['lesioned_endpoint_nostim'].mean(),
            'lesioned_nostim_endpoint_std': group['lesioned_endpoint_nostim'].std(),

            'endpoint_diff_mean': group['lesioned_endpoint_diff'].mean(),
            'endpoint_diff_std': group['lesioned_endpoint_diff'].std(),

            'lesioned_straightness_stim_mean': group['lesioned_straightness_stim'].mean(),
            'lesioned_straightness_stim_std': group['lesioned_straightness_stim'].std(),
            'lesioned_straightness_nostim_mean': group['lesioned_straightness_nostim'].mean(),
            'lesioned_straightness_nostim_std': group['lesioned_straightness_nostim'].std(),
            'straightness_diff_mean': group['lesioned_straightness_diff'].mean(),
            'straightness_diff_std': group['lesioned_straightness_diff'].std(),  # ADD THIS LINE

            'lesioned_maxdev_stim_mean': group['lesioned_maxdev_stim'].mean(),
            'lesioned_maxdev_stim_std': group['lesioned_maxdev_stim'].std(),
            'lesioned_maxdev_nostim_mean': group['lesioned_maxdev_nostim'].mean(),
            'lesioned_maxdev_nostim_std': group['lesioned_maxdev_nostim'].std(),
            'maxdev_diff_mean': group['lesioned_maxdev_diff'].mean(),
            'maxdev_diff_std': group['lesioned_maxdev_diff'].std(),  # ADD THIS LINE

            # ===== HEALTHY BEHAVIORAL METRICS =====
            'healthy_endpoint_stim_mean': group['healthy_endpoint_stim'].mean(),
            'healthy_endpoint_stim_std': group['healthy_endpoint_stim'].std(),
            'healthy_endpoint_nostim_mean': group['healthy_endpoint_nostim'].mean(),
            'healthy_endpoint_nostim_std': group['healthy_endpoint_nostim'].std(),
            'healthy_endpoint_diff_mean': group['healthy_endpoint_diff'].mean(),

            'healthy_straightness_stim_mean': group['healthy_straightness_stim'].mean(),
            'healthy_straightness_stim_std': group['healthy_straightness_stim'].std(),
            'healthy_straightness_nostim_mean': group['healthy_straightness_nostim'].mean(),
            'healthy_straightness_nostim_std': group['healthy_straightness_nostim'].std(),
            'healthy_straightness_diff_mean': group['healthy_straightness_diff'].mean(),

            'healthy_maxdev_stim_mean': group['healthy_maxdev_stim'].mean(),
            'healthy_maxdev_stim_std': group['healthy_maxdev_stim'].std(),
            'healthy_maxdev_nostim_mean': group['healthy_maxdev_nostim'].mean(),
            'healthy_maxdev_nostim_std': group['healthy_maxdev_nostim'].std(),
            'healthy_maxdev_diff_mean': group['healthy_maxdev_diff'].mean(),
        }

        # ===== jPCA METRICS - Total Variance for 4 conditions =====
        entry['jpca_total_var_lesioned_stim_mean'] = group['jpca_total_var_lesioned_stim'].mean()
        entry['jpca_total_var_lesioned_stim_std'] = group['jpca_total_var_lesioned_stim'].std()
        entry['jpca_total_var_lesioned_nostim_mean'] = group['jpca_total_var_lesioned_nostim'].mean()
        entry['jpca_total_var_lesioned_nostim_std'] = group['jpca_total_var_lesioned_nostim'].std()
        entry['jpca_total_var_healthy_stim_mean'] = group['jpca_total_var_healthy_stim'].mean()
        entry['jpca_total_var_healthy_stim_std'] = group['jpca_total_var_healthy_stim'].std()
        entry['jpca_total_var_healthy_nostim_mean'] = group['jpca_total_var_healthy_nostim'].mean()
        entry['jpca_total_var_healthy_nostim_std'] = group['jpca_total_var_healthy_nostim'].std()

        # ===== jPCA METRICS - All 5 Planes Variances =====
        for plane in range(1, 6):
            entry[f'jpca_plane{plane}_var_lesioned_stim_mean'] = group[f'jpca_plane{plane}_var_lesioned_stim'].mean()
            entry[f'jpca_plane{plane}_var_lesioned_stim_std'] = group[f'jpca_plane{plane}_var_lesioned_stim'].std()
            entry[f'jpca_plane{plane}_var_lesioned_nostim_mean'] = group[
                f'jpca_plane{plane}_var_lesioned_nostim'].mean()
            entry[f'jpca_plane{plane}_var_lesioned_nostim_std'] = group[f'jpca_plane{plane}_var_lesioned_nostim'].std()
            entry[f'jpca_plane{plane}_var_healthy_stim_mean'] = group[f'jpca_plane{plane}_var_healthy_stim'].mean()
            entry[f'jpca_plane{plane}_var_healthy_stim_std'] = group[f'jpca_plane{plane}_var_healthy_stim'].std()
            entry[f'jpca_plane{plane}_var_healthy_nostim_mean'] = group[f'jpca_plane{plane}_var_healthy_nostim'].mean()
            entry[f'jpca_plane{plane}_var_healthy_nostim_std'] = group[f'jpca_plane{plane}_var_healthy_nostim'].std()

        # ===== jPCA METRICS - All 5 Planes Rotational Frequency =====
        for plane in range(1, 6):
            entry[f'jpca_freq_plane{plane}_lesioned_stim_mean'] = group[f'jpca_freq_plane{plane}_lesioned_stim'].mean()
            entry[f'jpca_freq_plane{plane}_lesioned_stim_std'] = group[f'jpca_freq_plane{plane}_lesioned_stim'].std()
            entry[f'jpca_freq_plane{plane}_lesioned_nostim_mean'] = group[
                f'jpca_freq_plane{plane}_lesioned_nostim'].mean()
            entry[f'jpca_freq_plane{plane}_lesioned_nostim_std'] = group[
                f'jpca_freq_plane{plane}_lesioned_nostim'].std()
            entry[f'jpca_freq_plane{plane}_healthy_stim_mean'] = group[f'jpca_freq_plane{plane}_healthy_stim'].mean()
            entry[f'jpca_freq_plane{plane}_healthy_stim_std'] = group[f'jpca_freq_plane{plane}_healthy_stim'].std()
            entry[f'jpca_freq_plane{plane}_healthy_nostim_mean'] = group[
                f'jpca_freq_plane{plane}_healthy_nostim'].mean()
            entry[f'jpca_freq_plane{plane}_healthy_nostim_std'] = group[f'jpca_freq_plane{plane}_healthy_nostim'].std()

        summary.append(entry)

    return pd.DataFrame(summary)

def plot_amplitude_response(summary_df, save_path=None):
    """Plot endpoint error, straightness, and max deviation as function of amplitude for each timing."""
    timings = sorted(summary_df['timing'].unique())
    timing_labels = {-1.0: 't = -1.0 s', -0.2: 't = -0.2 s', 0: 't = 0 s',
                     0.2: 't = +0.2 s'}
    colors = plt.cm.viridis(np.linspace(0, 1, len(timings)))

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Store legend handles and labels for ALL timings
    legend_handles = []
    legend_labels = []
    seen_labels = set()  # To avoid duplicate legend entries

    # Track if healthy baseline has been added
    healthy_added = False

    for idx, timing in enumerate(timings):
        subset = summary_df[summary_df['timing'] == timing].sort_values('amplitude')
        color = colors[idx]
        amplitudes = subset['amplitude'].values

        # ===== Endpoint Error =====
        ax = axes[0, 0]

        # No stim line
        line_nostim, = ax.plot(amplitudes, subset['lesioned_nostim_endpoint_mean'],
                               'o--', color=color, alpha=0.5)
        # Stim line
        line_stim, = ax.plot(amplitudes, subset['lesioned_endpoint_mean'],
                             'o-', color=color, linewidth=2)

        # Add to legend (only once per timing)
        label_nostim = f'{timing_labels[timing]} (no stim)'
        label_stim = f'{timing_labels[timing]} (stim)'

        if label_nostim not in seen_labels:
            legend_handles.append(line_nostim)
            legend_labels.append(label_nostim)
            seen_labels.add(label_nostim)

        if label_stim not in seen_labels:
            legend_handles.append(line_stim)
            legend_labels.append(label_stim)
            seen_labels.add(label_stim)

        # Healthy baseline (add once)
        if not healthy_added:
            healthy_line = ax.axhline(y=subset['healthy_endpoint_nostim_mean'].mean(), color='black',
                                      linestyle=':', label='Healthy baseline (no stim)')
            legend_handles.append(healthy_line)
            legend_labels.append('Healthy baseline (no stim)')
            healthy_added = True
        else:
            ax.axhline(y=subset['healthy_endpoint_nostim_mean'].mean(), color='black',
                       linestyle=':')

        ax = axes[0, 1]
        ax.plot(amplitudes, subset['endpoint_diff_mean'],
                'o-', color=color, linewidth=2)
        ax.fill_between(amplitudes,
                        subset['endpoint_diff_mean'] - subset['endpoint_diff_std'],
                        subset['endpoint_diff_mean'] + subset['endpoint_diff_std'],
                        alpha=0.2, color=color)
        ax.axhline(y=0, color='gray', linestyle='--')

        # ===== Straightness =====
        ax = axes[1, 0]
        ax.plot(amplitudes, subset['lesioned_straightness_nostim_mean'],
                'o--', color=color, alpha=0.5)
        ax.plot(amplitudes, subset['lesioned_straightness_stim_mean'],
                'o-', color=color, linewidth=2)
        ax.axhline(y=subset['healthy_straightness_nostim_mean'].mean(), color='black',
                   linestyle=':')

        ax = axes[1, 1]
        ax.plot(amplitudes, subset['straightness_diff_mean'],
                'o-', color=color, linewidth=2)
        ax.fill_between(amplitudes,
                        subset['straightness_diff_mean'] - subset['straightness_diff_std'],
                        subset['straightness_diff_mean'] + subset['straightness_diff_std'],
                        alpha=0.2, color=color)
        ax.axhline(y=0, color='gray', linestyle='--')

        # ===== Max Deviation =====
        ax = axes[2, 0]
        ax.plot(amplitudes, subset['lesioned_maxdev_nostim_mean'],
                'o--', color=color, alpha=0.5)
        ax.plot(amplitudes, subset['lesioned_maxdev_stim_mean'],
                'o-', color=color, linewidth=2)
        ax.axhline(y=subset['healthy_maxdev_nostim_mean'].mean(), color='black',
                   linestyle=':')

        ax = axes[2, 1]
        ax.plot(amplitudes, subset['maxdev_diff_mean'],
                'o-', color=color, linewidth=2)
        ax.fill_between(amplitudes,
                        subset['maxdev_diff_mean'] - subset['maxdev_diff_std'],
                        subset['maxdev_diff_mean'] + subset['maxdev_diff_std'],
                        alpha=0.2, color=color)
        ax.axhline(y=0, color='gray', linestyle='--')

    # Formatting for all subplots
    for ax in axes.flatten():
        ax.set_xlabel('Stimulation Amplitude')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(summary_df['amplitude'].unique()))
        ax.set_xticklabels([int(x) if x == int(x) else x for x in sorted(summary_df['amplitude'].unique())])

    # Column titles
    axes[0, 0].set_ylabel('Endpoint Error (cm)')
    axes[0, 1].set_ylabel('ΔEndpoint Error')
    axes[1, 0].set_ylabel('Straightness Ratio')
    axes[1, 1].set_ylabel('ΔStraightness')
    axes[2, 0].set_ylabel('Max Deviation (cm)')
    axes[2, 1].set_ylabel('ΔMax Deviation')

    # Remove individual legends from all subplots
    for ax in axes.flatten():
        if ax.get_legend():
            ax.get_legend().remove()

    # Calculate number of columns for legend (2 columns per timing + healthy)
    n_legend_items = len(legend_handles)
    n_legend_cols = min(4, n_legend_items)  # At most 4 columns

    # Add a single legend above the first row of subplots
    fig.legend(legend_handles, legend_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=n_legend_cols,
               fontsize=8,
               frameon=True,
               fancybox=True,
               shadow=True)

    plt.suptitle('Systematic Stimulation Results - Behavioral', fontsize=14, y=1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_intensity_vs_peak_response(df, amplitude_col='amplitude',
                                    timing_col='timing', run_id_col='run_id',
                                    n_nearest=3, n_farthest=3,
                                    save_path=None):
    """
    Plot intensity-response curves for nearest and farthest neurons.
    Two subplots: Activity (nearest + farthest together) and Hidden (nearest + farthest together).
    """
    # Check if required columns exist
    sample_col = f'nearest_0_activity_peak'
    if sample_col not in df.columns:
        print("Warning: Per-neuron peak response columns not found.")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax in axes:
            ax.text(0.5, 0.5, 'Per-neuron data not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Data Not Available')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return fig

    timings = sorted(df[timing_col].unique())
    timing_labels = {-1.0: 'Preparation (t = -1.0 s)', -0.2: 't = -0.2 s', 0: 'Movement Onset (t = 0 s)',
                     0.2: 'Execution (t = +0.2 s)'}

    for timing in timings:
        subset = df[df[timing_col] == timing].copy()
        amplitudes = sorted(subset[amplitude_col].unique())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Colors
        nearest_colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_nearest))[::-1]
        farthest_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_farthest))[::-1]

        # Plot 1: Activity (nearest + farthest together)
        ax = axes[0]
        for i in range(n_nearest):
            peaks_by_amp = []
            stds_by_amp = []
            for amp in amplitudes:
                amp_subset = subset[subset[amplitude_col] == amp]
                peaks = amp_subset[f'nearest_{i}_activity_peak'].values
                peaks_by_amp.append(np.mean(peaks))
                stds_by_amp.append(np.std(peaks))
            dist = subset[f'nearest_{i}_dist'].iloc[0] if len(subset) > 0 else 0
            ax.errorbar(amplitudes, peaks_by_amp, yerr=stds_by_amp,
                        marker='o', capsize=3, markersize=6, linewidth=2,
                        color=nearest_colors[i], label=f'{i+1}. Nearest Neuron')

        for i in range(n_farthest):
            peaks_by_amp = []
            stds_by_amp = []
            for amp in amplitudes:
                amp_subset = subset[subset[amplitude_col] == amp]
                peaks = amp_subset[f'farthest_{i}_activity_peak'].values
                peaks_by_amp.append(np.mean(peaks))
                stds_by_amp.append(np.std(peaks))
            dist = subset[f'farthest_{i}_dist'].iloc[0] if len(subset) > 0 else 0
            ax.errorbar(amplitudes, peaks_by_amp, yerr=stds_by_amp,
                        marker='s', capsize=3, markersize=6, linewidth=2,
                        color=farthest_colors[i], label=f'{i+1}. Farthest Neuron',
                        linestyle='--')

        ax.set_xlabel('Stimulation Amplitude')
        ax.set_xticks(amplitudes)
        ax.set_ylabel('Peak Evoked Δ Neural Activity')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Hidden (nearest + farthest together)
        ax = axes[1]
        for i in range(n_nearest):
            peaks_by_amp = []
            stds_by_amp = []
            for amp in amplitudes:
                amp_subset = subset[subset[amplitude_col] == amp]
                peaks = amp_subset[f'nearest_{i}_hidden_peak'].values
                peaks_by_amp.append(np.mean(peaks))
                stds_by_amp.append(np.std(peaks))
            dist = subset[f'nearest_{i}_dist'].iloc[0] if len(subset) > 0 else 0
            ax.errorbar(amplitudes, peaks_by_amp, yerr=stds_by_amp,
                        marker='o', capsize=3, markersize=6, linewidth=2,
                        color=nearest_colors[i], label=f'{i+1}. Nearest Neuron')

        for i in range(n_farthest):
            peaks_by_amp = []
            stds_by_amp = []
            for amp in amplitudes:
                amp_subset = subset[subset[amplitude_col] == amp]
                peaks = amp_subset[f'farthest_{i}_hidden_peak'].values
                peaks_by_amp.append(np.mean(peaks))
                stds_by_amp.append(np.std(peaks))
            dist = subset[f'farthest_{i}_dist'].iloc[0] if len(subset) > 0 else 0
            ax.errorbar(amplitudes, peaks_by_amp, yerr=stds_by_amp,
                        marker='s', capsize=3, markersize=6, linewidth=2,
                        color=farthest_colors[i], label=f'{i+1}. Farthest Neuron',
                        linestyle='--')

        ax.set_xlabel('Stimulation Amplitude')
        ax.set_xticks(amplitudes)
        ax.set_ylabel('Peak Evoked Δ Activation State')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Intensity-Response Curves - {timing_labels.get(timing, f"t={timing}s")}', fontsize=14)
        plt.tight_layout()

        if save_path:
            timing_str = str(timing).replace('-', 'minus')
            save_path_timing = Path(save_path).parent / f'intensity_response_t{timing_str}.svg'
            plt.savefig(save_path_timing, bbox_inches='tight')
        plt.close()



def save_jpca_summary_table(summary_df, save_path=None):
    """Save jPCA summary table (mean and std of total variance for 4 conditions, plus across all conditions)."""

    # Define the 4 conditions
    conditions = [
        ('healthy_nostim', 'jpca_total_var_healthy_nostim_mean', 'jpca_total_var_healthy_nostim_std'),
        ('healthy_stim', 'jpca_total_var_healthy_stim_mean', 'jpca_total_var_healthy_stim_std'),
        ('lesioned_nostim', 'jpca_total_var_lesioned_nostim_mean', 'jpca_total_var_lesioned_nostim_std'),
        ('lesioned_stim', 'jpca_total_var_lesioned_stim_mean', 'jpca_total_var_lesioned_stim_std')
    ]

    # Collect mean and std for each condition
    results = []
    for cond_name, mean_col, std_col in conditions:
        if mean_col in summary_df.columns:
            mean_val = summary_df[mean_col].mean()
            std_val = summary_df[std_col].mean()  # Average of stds across conditions
            results.append({
                'Condition': cond_name.replace('_', ' ').title(),
                'Mean Total Variance': mean_val,
                'Std Total Variance': std_val
            })

    # Add row for "across all conditions"
    all_means = [r['Mean Total Variance'] for r in results]
    all_stds = [r['Std Total Variance'] for r in results]
    results.append({
        'Condition': 'Across All Conditions',
        'Mean Total Variance': np.mean(all_means),
        'Std Total Variance': np.mean(all_stds)
    })

    # Create DataFrame
    jpca_table = pd.DataFrame(results)

    # Save to CSV
    if save_path is None:
        save_path = Path('jpca_summary_table.csv')
    jpca_table.to_csv(save_path, index=False)

    # Also print to console
    print("\n" + "=" * 60)
    print("jPCA SUMMARY TABLE")
    print("=" * 60)
    print(jpca_table.to_string(index=False))
    print(f"\nTable saved to {save_path}")

    return jpca_table


def main(name):
    # Define paths (matches run_systematic_experiments.py)
    results_dir = Path.cwd() / 'results' / name / 'systematic_results'
    csv_path = results_dir / 'systematic_results.csv'

    # Check if file exists
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        print("Please run run_systematic_experiments.py first.")
        return

    # Load and compute
    df = load_and_compute_metrics(csv_path)
    summary_df = compute_summary_stats(df)

    # Save summary in the same directory
    summary_df.to_csv(results_dir / 'summary_results.csv', index=False)
    print(f"Summary saved to {results_dir / 'summary_results.csv'}")

    # Create output directory for plots
    output_dir = results_dir / 'analysis_plots'
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_amplitude_response(summary_df, save_path=output_dir / 'amplitude_response.svg')

    # Generate intensity-response curves (using original df, not summary_df)
    plot_intensity_vs_peak_response(df, save_path=output_dir / 'intensity_response.svg')

    # Save jPCA summary table (instead of printing)
    save_jpca_summary_table(summary_df, save_path=results_dir / 'jpca_summary_table.csv')

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main(name='test_2')    # TODO