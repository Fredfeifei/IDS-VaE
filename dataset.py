import json
import torch
from torch.utils.data import Dataset, DataLoader
import config

class MyIndexedDataset(Dataset):
    def __init__(self, data_file_path, index_file_path, img_size=16, channels=20):
        self.data_file_path = data_file_path
        self.index_file_path = index_file_path
        self.img_size = img_size
        self.channels = channels

        with open(index_file_path, 'r') as f:
            self.indexes = [int(line.strip()) for line in f]
        self.length = len(self.indexes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        offset = self.indexes[idx]
        with open(self.data_file_path, 'r', encoding='utf-8') as file:
            file.seek(offset)
            line = file.readline().strip()
            bson_ds = json.loads(line)

        label = bson_ds['label']
        sparse_matrix = bson_ds['sparse_matrix']
        sparse_matrix = torch.tensor(sparse_matrix, dtype=torch.long)

        return sparse_matrix, sparse_matrix, label

def create_dataloaders(train_file, train_index, test_file, test_index, batch_size=config.batch_size):
    dataset_train = MyIndexedDataset(data_file_path=train_file, index_file_path=train_index)
    dataset_test = MyIndexedDataset(data_file_path=test_file, index_file_path=test_index)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_test

