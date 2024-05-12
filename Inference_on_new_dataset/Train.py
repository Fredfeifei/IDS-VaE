from dataset import create_dataloaders
from model_VaE import VAE
from model_classifier import NewDecoder
import torch
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score
import os

model = VAE()
model_dir = ".\model_para\model_epoch_VaE_New_classifier.pth"
model_weights = torch.load(model_dir)
encoder_weights = {k.replace('encoder.', ''): v for k, v in model_weights.items() if k.startswith('encoder.')}
model.encoder.load_state_dict(encoder_weights, strict=False)

if config.freeze:
    for param in model.encoder.parameters():
        param.requires_grad = False
    
model.decoder = NewDecoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00015)
num_epochs = 150
save_model = "./model_para"
model = model.to(device)

dataloader_train, dataloader_test = create_dataloaders(config.train_file_dir_2017, config.train_index_dir_2017, config.test_file_dir_2017, config.test_index_dir_2017)

for epoch in range(num_epochs):

    print(f"Working on {epoch}!")
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    model.train()
    for sparse_matrix, labels in dataloader_train:

        labels = labels.to(device)
        sparse_matrix = sparse_matrix.to(device)
        sparse_matrix = torch.nn.functional.one_hot(sparse_matrix.long(), num_classes=256).to(torch.float32)

        outputs = model(sparse_matrix)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_labels = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted_labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader_test)
    epoch_f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
    epoch_accuracy = accuracy_score(all_labels, all_predictions) * 100

    print(f"Training Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%, F1 Weighted: {epoch_f1_weighted:.4f}, F1 Normal: {epoch_f1:.4f}")

    if (epoch + 1) % 30 == 0:
        save_dir = os.path.join(save_model, f'model_epoch_{epoch+1}.pth')
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(model.state_dict(), save_dir)
        print(f'Model saved at epoch {epoch+1}')
        
        model.eval()
        val_labels = []
        val_predictions = []
        with torch.no_grad():
            for sp, label in dataloader_test:
                label = label.to(device)
                sp = sp.to(device)
                sp = torch.nn.functional.one_hot(sp.long(), num_classes=256).float()

                outputs = model(sp)
                _, predicted = torch.max(outputs, 1)

                val_labels.extend(label.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        test_f1_weighted = f1_score(val_labels, val_predictions, average='weighted')
        test_f1 = f1_score(val_labels, val_predictions, average='macro')
        test_accuracy = accuracy_score(val_labels, val_predictions) * 100

        print(f"Test Metrics Epoch [{epoch+1}/{num_epochs}], Accuracy: {test_accuracy:.4f}%, F1 Weighted: {test_f1_weighted:.4f}, F1 Normal: {test_f1:.4f}")