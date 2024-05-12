import torch
import torch.optim as optim
import os
import config

def train_model(model, dataloader, num_epochs=15, lr=0.0002, save_model_dir=config.save_model_dir):
    model = model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Using {device} for training.")

    for epoch in range(num_epochs):
        print(f"Working on {epoch}!")
        running_loss = 0.0
        model.train()

        for sparse_matrix, labels, _ in dataloader:
            labels = labels.to(device).float()
            sparse_matrix = sparse_matrix.to(device)
            sparse_matrix = torch.nn.functional.one_hot(sparse_matrix.long(), num_classes=256).float()

            outputs, mu, var = model(sparse_matrix)
            loss = model.loss_function(outputs, labels, mu, var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        print(f"Training Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_model_dir, f'model_epoch_{epoch+1}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at epoch {epoch+1}')

def test_model(model, dataloader_test):
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss_test = 0.0

    with torch.no_grad():
        for sparse_matrix, labels, _ in dataloader_test:
            labels = labels.to(device)
            sparse_matrix = sparse_matrix.to(device)
            sparse_matrix = torch.nn.functional.one_hot(sparse_matrix.long(), num_classes=256).float()

            outputs, mu, var = model(sparse_matrix)
            loss = model.loss_function(outputs, labels, mu, var)

            running_loss_test += loss.item()

    epoch_loss_test = running_loss_test / len(dataloader_test)
    print(f"Test Metrics Loss: {epoch_loss_test:.4f}")