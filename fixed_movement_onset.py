import numpy as np
from pathlib import Path
from dataset import create_data_velocity_centeroutreach_fixed_movement_onset


def main(name='test'):

    directory = Path.cwd() / 'results' / name
    params = np.load(directory / 'params.npy', allow_pickle=True).item()
    data = create_data_velocity_centeroutreach_fixed_movement_onset(params['data'])
    np.save(directory / 'fixed_movement_onset_data', data)


if __name__ == "__main__":
    main()
