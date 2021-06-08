import numpy as np


def add_noise(data: np.ndarray, feature_labels: np.ndarray, noisy_feats: int) -> np.ndarray:
    """
    Adds random normal noise to selected number of features/columns.

    :param data: the dataset (as an array of arrays) to perform the manipulation on.
    :param feature_labels: np.ndarray of the feature labels.
    :param noisy_feats: integer of features to add noise to.
    :return: the new data after manipulation
    """
    # extract some measures, randomly select the columns to add noise on
    dim = len(data[0])
    rng = np.random.default_rng()
    noisy_indices = rng.choice(dim, size=noisy_feats, replace=False)
    noisy_labels = []
    # this part is only for printout
    for i in noisy_indices:
        noisy_labels.append(feature_labels[i])
    print('Adding Noise to {}'.format(noisy_labels))
    # actual noise being added
    for i in noisy_indices:
        mean = np.mean(data[:, i])
        std = np.std(data[:, i])
        # need variables for random noise to be added
        data[:, i] += np.random.normal(mean, std, len(data))
    return data
