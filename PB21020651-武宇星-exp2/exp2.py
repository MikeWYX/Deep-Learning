import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, random_split, DataLoader
epochs = 25
learning_rate = 2e-3
drop = 0.2
batch_size = 1024
gamma = 0.1
step_size = 10
device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
dataset = torchvision.datasets.CIFAR10(root='//data/wuyux', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='//data/wuyux', train=False, download=False, transform=transform )
dataset_size = len(dataset)
# image_size = dataset[0].size()
# print("Image size:", image_size)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding="same")
        self.conv2 = nn.Conv2d(64, 64, 3, padding="same")
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv4 = nn.Conv2d(128, 128, 3, padding="same")
        self.conv5 = nn.Conv2d(128, 256, 3, padding="same")
        self.conv6 = nn.Conv2d(256, 256, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        x = (self.relu(self.conv1(x)))
        x = self.maxpool(self.bn1(self.relu(self.conv2(x))))
        x = (self.relu(self.conv3(x)))
        x = self.maxpool(self.bn2(self.relu(self.conv4(x))))
        x = (self.relu(self.conv5(x)))
        x = self.avgpool(self.bn3(self.relu(self.conv6(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # x = self.fc3(x)
        return x

model = CNN()
model.to(device)
# print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()  
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    train_losses.append(train_loss / len(train_loader))
    train_accuracy = correct / total
    
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted_val = outputs.max(1)
            total_val += targets.size(0)
            correct_val += predicted_val.eq(targets).sum().item()
        val_losses.append(val_loss / len(val_loader))
        val_accuracy = correct_val / total_val
    scheduler.step()
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Train Loss', color='blue')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('result/train_loss_conv_out_32_64.png')
# plt.close()

# calculate accuracy on test set
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted_test = outputs.max(1)
        total_test += targets.size(0)
        correct_test += predicted_test.eq(targets).sum().item()
    test_accuracy = correct_test / total_test
    print(f'Test Accuracy: {test_accuracy:.4f}')
