import numpy as np
import matplotlib.pyplot as plt


def plot_transform(transform_type='reciprocal', **kwargs):
    """
    Quick plot of a single transform with adjustable parameters.

    transform_type: 'reciprocal', 'gaussian', or 'linear'
    kwargs: parameters for the specific transform
    """
    distances = np.linspace(0, np.sqrt(2)/2, 1000)

    if transform_type == 'reciprocal':
        max_distance = kwargs.get('max_distance', 0.1)
        power = kwargs.get('power', 1)
        weights = 1 / (1 + (distances / max_distance) ** power)
        title = f'Reciprocal Transform (d₀={max_distance}, p={power})'

    elif transform_type == 'gaussian':
        sigma = kwargs.get('sigma', 0.2)
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
        title = f'Gaussian Transform (σ={sigma})'

    elif transform_type == 'linear':
        max_distance = kwargs.get('max_distance', 0.5)
        weights = np.maximum(0, 1 - distances / max_distance)
        title = f'Linear Transform (d_max={max_distance})'

    else:
        raise ValueError("transform_type must be 'reciprocal', 'gaussian', or 'linear'")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, weights, linewidth=2, color='blue')
    ax.fill_between(distances, weights, alpha=0.3)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Weight')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, np.sqrt(2)/2)

    # Mark half-maximum point
    half_max_idx = np.argmin(np.abs(weights - 0.5))
    ax.plot(distances[half_max_idx], 0.5, 'ro', markersize=8)
    
    ax.annotate(f'd₅₀ = {distances[half_max_idx]:.3f}',
                xy=(distances[half_max_idx], 0.5),
                xytext=(distances[half_max_idx] + 0.1, 0.6),
                arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.show()
    return fig, ax


# Examples:
if __name__ == "__main__":
    # Reciprocal with different parameters
    plot_transform('reciprocal', max_distance=0.1, power=1)
    plot_transform('reciprocal', max_distance=0.04, power=1)
    plot_transform('reciprocal', max_distance=0.2, power=1)
    '''
    # Gaussian with different sigma
    plot_transform('gaussian', sigma=0.15)
    plot_transform('gaussian', sigma=0.2)
    plot_transform('gaussian', sigma=0.3)

    # Linear with different max distance
    plot_transform('linear', max_distance=0.3)
    plot_transform('linear', max_distance=0.5)
    plot_transform('linear', max_distance=0.7)
    '''