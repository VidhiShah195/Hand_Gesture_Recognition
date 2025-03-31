import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class GestureModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GestureModel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size), 
            nn.ReLU(),
            nn.Dropout(0.2) 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(hidden_size, output_size) 
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)

class GestureDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.labels = []
        
        gesture_labels = {
            "open_palm": 0,  # No filter
            "fist": 1,  # Grayscale
            "peace_sign": 2,  # Sepia
            "thumbs_up": 3,  # Blur
            "pointing_finger": 4,  # Edge detection
            "ok_sign": 5  # Cartoon filter
        }

        for gesture_name in os.listdir(data_path):
            if gesture_name.endswith('.csv'):
                print(f"Processing file: {gesture_name}")
                
                gesture_base_name = gesture_name.split('.')[0]
                
                if gesture_base_name in gesture_labels:
                    label = gesture_labels[gesture_base_name]
                else:
                    print(f"Skipping file due to unknown gesture: {gesture_name}")
                    continue

                file_path = os.path.join(data_path, gesture_name)
                df = pd.read_csv(file_path, header=None)
                
                for i in range(df.shape[0]):
                    self.data.append(df.iloc[i, 1:].values) 
                    self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

input_size = 42 
hidden_size = 128
output_size = 6
batch_size = 32
learning_rate = 0.001
epochs = 30
dropout_rate = 0.2  

data_path = 'gesture_data'
dataset = GestureDataset(data_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = GestureModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    for data, labels in train_loader:
        outputs = model(data)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct_preds / total_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    val_loss = total_loss / len(val_loader)
    val_accuracy = 100 * correct_preds / total_samples
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    scheduler.step()

torch.save(model.state_dict(), 'gesture_model.pth')
print("Model saved as gesture_model_improved.pth")

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
