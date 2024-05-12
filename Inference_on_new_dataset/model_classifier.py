import torch.nn as nn
import config

class NewDecoder(nn.Module):
    def __init__(self, input_features = 128, num_classes = config.num_class_2017):
        super(NewDecoder, self).__init__()
        self.num_class = num_classes
        self.layers = nn.Sequential(
             nn.Linear(128, 64),
             nn.BatchNorm1d(64),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)