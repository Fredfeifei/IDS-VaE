from dataset import create_dataloaders
from model_VaE import VAE
from train import train_model, test_model
import config

# Dataset paths
train_file_dir = config.train_file_dir_2018
test_file_dir = config.test_file_dir_2018
train_index_dir = config.train_index_dir_2018
test_index_dir = config.test_index_dir_2018

# Dataloaders
dataloader_train, dataloader_test = create_dataloaders(train_file_dir, train_index_dir, test_file_dir, test_index_dir)
model = VAE()

train_model(model, dataloader_train, dataloader_test)
test_model(model, dataloader_test)