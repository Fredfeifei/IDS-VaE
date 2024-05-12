import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from dataset import create_dataloaders
from model_VaE import VAE
from model_classifier import NewDecoder
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = ".\model_para\model_epoch_VaE_New_classifier.pth"
model = VAE(is_train=False)
model.decoder = NewDecoder()
model = model.to(device)
model.load_state_dict(torch.load(model_dir, map_location=device))

# Data Loaders
dataloader_train, dataloader_test = create_dataloaders(
    config.train_file_dir_2017, 
    config.train_index_dir_2017, 
    config.test_file_dir_2017, 
    config.test_index_dir_2017
)

# Evaluation: test:80% of CIC-IDS2017, train: 20% of CIC-IDS2017
val_labels = []
val_predictions = []

for sp, _, label in dataloader_test:
    sp = sp.to(device)
    label = label.to(device)
    sp = F.one_hot(sp.long(), num_classes=256).float()

    with torch.no_grad():
        outputs = model(sp)
        _, predicted = torch.max(outputs[0], 1)

    val_labels.extend(label.cpu().numpy())
    val_predictions.extend(predicted.cpu().numpy())

test_f1_weighted = f1_score(val_labels, val_predictions, average='weighted')
test_f1 = f1_score(val_labels, val_predictions, average='macro')
test_accuracy = accuracy_score(val_labels, val_predictions) * 100

print(f"Accuracy: {test_accuracy:.4f}%, F1 Weighted: {test_f1_weighted:.4f}, F1 Normal: {test_f1:.4f}")