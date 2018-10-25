from abc import ABCMeta
import numpy as np
import pandas as pd
import glob
from torch.utils.data import Dataset
import torch.tensor


class Parser(metaclass=ABCMeta):
    def __init__(self, extension, sep):
        self.extension = extension
        self.sep = sep

    def register_data(self, folder):
        pass

    def get_data(self, folder, idx):
        pass


class FileParser(Parser):
    def parse_string(self, fs):
        data = []
        for token in fs.split(self.sep):
            try:
                data.append(float(token.strip()))
            except ValueError:
                continue
        return data

    def parse_filename(self, file):
        return file.split('/')[-1].split('.')[0]

    def get_filename_for_idx(self, folder, idx):
        return folder + idx + self.extension

    def get_data(self, folder, idx):
        filename = self.get_filename_for_idx(folder, idx)
        with open(filename, 'r') as fs:
            data = self.parse_string(fs.read())
        return data

    def get_files_list(self, path):
        return glob.glob(path + '*')

    def register_data(self, path):
        names = map(lambda file: self.parse_filename(file), self.get_files_list(path))
        return list(names)


class DataSampler(Dataset):
    def __init__(self, parser, folder):
        self.parser = parser
        data = self.process_data(folder)
        self.data, self.data_idx = data.values, data.index
        self.encoded_names = self.encode_names(self.data_idx)
        self.labels = np.array(list(map(lambda x: x.split('_')[0], self.data_idx)))
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

    def process_data(self, folder):
        data = []
        names = []
        files = self.parser.get_files_list(folder)
        for file in files:
            with open(file, 'r') as fs:
                x = self.parser.parse_string(fs.read())
                name = self.parser.parse_filename(file)
                data.append(x)
                names.append(name)
        return pd.DataFrame(data, index=names)

    def encode_names(self, names):
        from sklearn.preprocessing import LabelEncoder
        return LabelEncoder().fit_transform(names)

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.data[index], self.labels[index]
        if target == 0:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2 = self.data[siamese_index]
        return (img1, img2), target

    def __len__(self):
        return len(self.data_idx)


class SiameseData(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, parser, folder):
        self.folder = folder
        self.data_idx = parser.register_data(folder)
        self.parser = parser

    def get_positive_indices(self, idx, sep='_'):
        return list(filter(lambda x: x.split(sep)[0] == idx.split(sep)[0], self.data_idx))

    def get_negative_indices(self, idx, sep='_'):
        return list(filter(lambda x: x.split(sep)[0] != idx.split(sep)[0], self.data_idx))

    def sample_positive_index(self, idx, sep='_'):
        return np.random.choice(self.get_positive_indices(idx, sep))

    def sample_negative_index(self, idx, sep='_'):
        return np.random.choice(self.get_negative_indices(idx, sep))

    def sample_indices(self, index, mode='contrastive'):
        idx = self.data_idx[index]
        if mode == 'contrastive':
            target = np.random.randint(0, 2)
            # sample positive
            if target == 0:
                positive_index = idx
                while positive_index == idx:
                    positive_index = self.sample_positive_index(idx)
                return (idx, positive_index), target
            else:
                negative_index = self.sample_negative_index(idx)
                return (idx, negative_index), target
        elif mode == 'triplet':
            positive_index = self.sample_positive_index(idx)
            negative_index = self.sample_negative_index(idx)
            return idx, positive_index, negative_index

    def __getitem__(self, index):
        pair, target = self.sample_indices(index, mode='contrastive')
        x0 = torch.tensor(self.parser.get_data(self.folder, pair[0]))
        x1 = torch.tensor(self.parser.get_data(self.folder, pair[1]))
        return (x0, x1), target

    def __len__(self):
        return len(self.data_idx)
