import numpy as np


def add_noise(data: np.ndarray, feature_labels: np.ndarray, noisy_feats: int,
              noise: int, random_state: int = None) -> np.ndarray:
    """
    Adds random normal noise to selected number of features/columns.

    :param data: the dataset (as an array of arrays) to perform the manipulation on.
    :param feature_labels: np.ndarray of the feature labels.
    :param noisy_feats: integer of features to add noise to.
    :param noise: standard deviation of noise to add.
    :param random_state: the seed for the random number generator
    :return: the new data after manipulation
    """
    # extract some measures, randomly select the columns to add noise on
    dim = len(data[0])
    rng = np.random.default_rng(random_state)
    noisy_indices = rng.choice(dim, size=noisy_feats, replace=False)
    noisy_labels = []
    # this part is only for printout
    for i in noisy_indices:
        noisy_labels.append(feature_labels[i])
    print('Adding Noise to {}'.format(noisy_labels))
    # actual noise being added
    for i in noisy_indices:
        data[:, i] = np.random.normal(0, noise, len(data[:, i]))
    return data
