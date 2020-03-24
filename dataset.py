# coding: UTF-8
"""This module defines an example Torch dataset from the Oze datachallenge.

Example
-------
$ dataloader = DataLoader(OzeDataset(DATSET_PATH),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

"""
import datetime
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OzeEvaluationDataset(Dataset):
    """Torch dataset for Oze datachallenge evaluation.

    Load dataset from two train.csv and test.csv file.

    Attributes
    ----------
    x: np.array
        Dataset input of shape (m, K, 37).
    labels: Dict
        Ordered labels list for R, Z and X.
    m: np.array
        Normalization constant.
    M: np.array
        Normalization constant.
    """

    def __init__(self, dataset_x_path, labels_path="labels.json", add_time=False, **kwargs):
        """Load dataset from csv.

        Parameters
        ---------
        dataset_x_path: str or Path
            Path to the dataset inputs as csv.
        labels_path: str or Path, optional
            Path to the labels, divided in R, Z and X, in json format.
            Default is "labels.json".
        """
        super().__init__(**kwargs)
        self.add_time = add_time
        self._load_x_from_csv(dataset_x_path, labels_path)

    def _load_x_from_csv(self, dataset_x_path, labels_path):
        """Load input dataset from csv and create x_train tensor."""
        # Load dataset as csv
        x = pd.read_csv(dataset_x_path)

        # Load labels, file can be found in challenge description
        with open(labels_path, "r") as stream_json:
            self.labels = json.load(stream_json)

        m = x.shape[0]
        K = 672  # Can be found through csv

        # Create R and Z
        R = x[self.labels["R"]].values
        R = np.tile(R[:, np.newaxis, :], (1, K, 1))

        Z = x[[f"{var_name}_{i}" for var_name in self.labels["Z"]
               for i in range(K)]]
        Z = Z.values.reshape((m, -1, K))
        Z = Z.transpose((0, 2, 1))

        '''add features'''
        if self.add_time==True:
            # get time(day, month, year)
            time = R[:,0,-3:] # shape 7500,672
            days = np.zeros((m, K))
            for i in range(m):
                days[i] = np.mod(np.arange(K)//24 + 
                                 datetime.date(int(time[i, 2]), 
                                               int(time[i, 1]), 
                                               int(time[i, 0])
                                               ).weekday(), 7)
            R = np.concatenate((R, days.reshape(m, K ,1)), axis=2)
            self.days = days
        # Store R and Z as x_train
        self._x = np.concatenate([Z, R], axis=-1)
        # Normalize
        self.M = np.max(self._x, axis=(0, 1))
        self.m = np.min(self._x, axis=(0, 1))
        self._x = (self._x - self.m) / (self.M - self.m + np.finfo(float).eps)
        # Convert to float32
        self._x = self._x.astype(np.float32)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._x[idx]

    def __len__(self):
        return self._x.shape[0]


class OzeDataset(OzeEvaluationDataset):
    """Torch dataset for Oze datachallenge training.

    Attributes
    ----------
    dataset_y_path: str or Path
        Path to the dataset targets as csv.
    y: np.array
        Dataset target of shape (m, K, 8).
    """

    def __init__(self, dataset_x_path, dataset_y_path, labels_path="labels.json", normalize=False, **kwargs):
        """Load dataset from csv.

        Parameters
        ---------
        dataset_x_path: str or Path
            Path to the dataset inputs as csv.
        dataset_y_path: str or Path
            Path to the dataset targets as csv.
        labels_path: str or Path, optional
            Path to the labels, divided in R, Z and X, in json format.
            Default is "labels.json".
        """
        super().__init__(dataset_x_path=dataset_x_path, labels_path=labels_path, **kwargs)
        self.normalize = normalize
        self._load_y_from_csv(dataset_y_path)

    def _load_y_from_csv(self, dataset_y_path):
        """Load target dataset from csv and create y_train tensor."""
        # Load dataset as csv
        y = pd.read_csv(dataset_y_path)

        m = y.shape[0] # size of dataset
        K = 672  # Can be found through csv

        # Create X
        X = y[[f"{var_name}_{i}" for var_name in self.labels["X"]
               for i in range(K)]]
        X = X.values.reshape((m, -1, K))
        X = X.transpose((0, 2, 1))

        # Store X as y_train
        self._y = X
        # Normalize
        if self.normalize:
            self.M = np.max(self._y, axis=(0, 1))
            self.m = np.min(self._y, axis=(0, 1))
            self._y = (self._y - self.m) / (self.M - self.m + np.finfo(float).eps)
        # Convert to float32
        self._y = self._y.astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self._x[idx], self._y[idx])


class OzeNPZDataset(Dataset):
    """Torch dataset for Oze datachallenge.

    Load dataset from a single npz file.

    Attributes
    ----------
    x: np.array
        Dataset input of shape (m, K, 37).
    y: np.array
        Dataset target of shape (m, K, 8).
    labels: Dict
        Ordered labels list for R, Z and X.
    m: np.array
        Normalization constant.
    M: np.array
        Normalization constant.
    """

    def __init__(self, dataset_path, labels_path="labels.json", **kwargs):
        """Load dataset from npz.

        Parameters
        ---------
        dataset_x: str or Path
            Path to the dataset inputs as npz.
        labels_path: str or Path, optional
            Path to the labels, divided in R, Z and X, in json format.
            Default is "labels.json".
        """
        super().__init__(**kwargs)

        self._load_npz(dataset_path, labels_path)

    def _load_npz(self, dataset_path, labels_path):
        """Load dataset from csv and create x_train and y_train tensors."""
        # Load dataset as csv
        dataset = np.load(dataset_path)

        # Load labels, can be found through csv or challenge description
        with open(labels_path, "r") as stream_json:
            self.labels = json.load(stream_json)

        R, X, Z = dataset['R'], dataset['X'], dataset['Z']
        m = Z.shape[0]  # Number of training example
        K = Z.shape[-1]  # Time serie length

        R = np.tile(R[:, np.newaxis, :], (1, K, 1))
        Z = Z.transpose((0, 2, 1))
        X = X.transpose((0, 2, 1))

        # Store R, Z and X as x_train and y_train
        self._x = np.concatenate([Z, R], axis=-1)
        # Normalize
        M = np.max(self._x, axis=(0, 1))
        m = np.min(self._x, axis=(0, 1))
        self._x = (self._x - m) / (M - m + np.finfo(float).eps)
        # Convert to float32
        self._x = self._x.astype(np.float32)

        self._y = X
        self.original_y = np.array(self._y).astype(np.float32)
        # Normalize
        self.M = np.max(self._y, axis=(0, 1))
        self.m = np.min(self._y, axis=(0, 1))
        self._y = (self._y - m) / (M - m + np.finfo(float).eps)
        # Convert to float32
        self._y = self._y.astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self._x[idx], self._y[idx])

    def __len__(self):
        return self._x.shape[0]
