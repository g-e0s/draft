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
        self.labels = np.array(list(map(lambda x: x.split('_')[0], self.data_idx)))
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

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

    def sample_positive_index(self, label):
        positive_index = np.random.choice(self.label_to_indices[label])
        return positive_index

    def sample_negative_index(self, label):
        random_label = np.random.choice(list(self.labels_set - {label}))
        negative_index = np.random.choice(self.label_to_indices[random_label])
        return negative_index

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        x0, label = self.data[index], self.labels[index]
        if target == 0:
            siamese_index = index
            while siamese_index == index:
                siamese_index = self.sample_positive_index(label)
        else:
            siamese_index = self.sample_negative_index(label)
        x1 = self.data[siamese_index]
        return (torch.tensor(x0).float(), torch.tensor(x1).float()), target

    def __len__(self):
        return len(self.data_idx)


class SiameseData(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, parser, folder):
        from sklearn.preprocessing import LabelEncoder
        self.parser = parser
        self.dataset = self.process_data(folder)
        self.encoder = LabelEncoder()
        self.true_labels = self.dataset.index
        self.labels = self.encoder.fit_transform(self.true_labels)
        self.train_data = self.dataset.values
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

    def process_data(self, folder):
        data_array = []
        data_idx = []
        for file in self.parser.get_files_list(folder):
            with open(file, 'r') as fs:
                data_arr = [float(x) for x in fs.read().strip().split(' ')[1:-1]]
                _id = file.split('/')[-1].split('_')[0]
                data_array.append(data_arr)
                data_idx.append(_id)
        return pd.DataFrame(data_array, index=data_idx)

    def sample_positive_index(self, label):
        positive_index = np.random.choice(self.label_to_indices[label])
        return positive_index

    def sample_negative_index(self, label):
        random_label = np.random.choice(list(set(self.label_to_indices.keys()) - {label}))
        negative_index = np.random.choice(self.label_to_indices[random_label])
        return negative_index

    def get_all_pairs(self, index):
        x0, label = self.train_data[index], self.labels[index].item()
        positives = np.vstack([self.train_data[idx] for idx in self.label_to_indices[label] if idx != label])
        negatives = np.vstack([self.train_data[idx] for idxs in list(self.labels_set - {label}) for idx in self.label_to_indices[idxs]])
        return (torch.tensor(np.tile(x0, (len(positives) + len(negatives), 1))).float(),
                torch.tensor(np.vstack([positives, negatives])).float()),\
               np.hstack([np.repeat(1, len(positives)), np.repeat(0, len(negatives))])

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        x0, label1 = self.train_data[index], self.labels[index].item()
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        x1 = self.train_data[siamese_index]

        return (torch.tensor(x0).float(), torch.tensor(x1).float()), target

    def __len__(self):
        return len(self.dataset)


class EncodedData(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, siamese_loader, model):
        self.siamese_loader = siamese_loader
        self.model = model

    def __getitem__(self, index):
        (x0, x1), target = self.siamese_loader.__getitem__(index)
        x0, x1 = self.model(x0, x1)
        return (x0, x1), target

    def __len__(self):
        return len(self.siamese_loader.dataset)

