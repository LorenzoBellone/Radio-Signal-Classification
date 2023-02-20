from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import h5py


class MyDataSet(Dataset):

    def __init__(self, hdf5_path: str, mod_class: list, indexes: list, model='convnext', transform=None):
        self.hdf5_path = hdf5_path
        self.indexes = indexes
        self.mod_class = mod_class
        self.transform = transform
        self.model = model

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):

        with h5py.File(self.hdf5_path, 'r') as f:
            x = f['X']
            index = self.indexes[item]
            signal = x[index]

        if "convnet" in self.model:
            signal1 = np.reshape(signal[:, 0], [32, 32])
            signal1 = np.expand_dims(signal1, axis=2)
            signal2 = np.reshape(signal[:, 1], [32, 32])
            signal2 = np.expand_dims(signal2, axis=2)
            signal = np.concatenate((signal1, signal2), axis=2)

        label = self.mod_class[item]
        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label

class MyHisarDataSet(Dataset):

    def __init__(self, hdf5_path: str, labels_path: list, indexes: list, model='convnext', transform=None):
        self.df_test = pd.read_hdf(hdf5_path)
        self.indexes = indexes
        self.mod_class = pd.read_csv(labels_path, header=None)
        with open('../HisarMod2019.1/class_indices.json', 'r') as f:
            self.class_idx = json.load(f)
        self.transform = transform
        self.model = model

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        
        if len(self.mod_class) == len(self.df_test):
            index = self.indexes[item]
            sample = self.df_test.iloc[index]
        else:
            sample = self.df_test.iloc[item]

        real = sample.apply(lambda x: np.single(x.real))
        imag = sample.apply(lambda x: np.single(x.imag))

        signal = np.ones((1, 1024, 2), dtype=np.single)
        signal[0, :, 0] = real
        signal[0, :, 1] = imag

        if "convnet" in self.model:
            signal1 = np.reshape(signal[:, 0], [32, 32])
            signal1 = np.expand_dims(signal1, axis=2)
            signal2 = np.reshape(signal[:, 1], [32, 32])
            signal2 = np.expand_dims(signal2, axis=2)
            signal = np.concatenate((signal1, signal2), axis=2)
        
        label = self.mod_class.iloc[self.indexes[item]].item()
        label = int(self.class_idx[str(label)])
        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label
