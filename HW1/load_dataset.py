import torch
from torch.utils.data import TensorDataset


def load_dataset(dataset_path, mean_subtraction, normalization):
    """
    Reads the train and validation data

    Arguments
    ---------
    dataset_path: (string) representing the file path of the dataset
    mean_subtraction: (boolean) specifies whether to do mean centering or not. Default: False
    normalization: (boolean) specifies whether to normalizes the data or not. Default: False

    Returns
    -------
    train_ds (TensorDataset): The features and their corresponding labels bundled as a dataset
    """
    # Load the dataset and extract the features and the labels
    dataset = torch.load(dataset_path)
    features = dataset['features']
    labels = dataset['labels']

    # Do mean_subtraction if it is enabled
    if mean_subtraction:
        mean = torch.mean(features, axis=0)
        features = torch.sub(features, mean)

    # do normalization if it is enabled
    if normalization:
        std = torch.std(features, axis=0)
        nonzero = (std != 0)
        features[:, nonzero] = torch.div(features[:, nonzero],  std[nonzero])

    # create tensor dataset train_ds
    train_ds = TensorDataset(features, labels)

    return train_ds