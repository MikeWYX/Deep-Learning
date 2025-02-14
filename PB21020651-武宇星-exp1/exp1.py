import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split, DataLoader

N = 10000 
width = 200
depth = 9
learning_rate = 0.00006
epochs = 100

# 数据集生成
x = torch.linspace(1, 16, steps = N).unsqueeze(1)  # 生成输入数据
y = torch.log2(x) + torch.cos(torch.pi * x / 2)  # 计算目标输出
dataset = TensorDataset(x, y)
# 划分数据集
train_size = int(N * 0.8)
val_size = int(N * 0.1)
test_size = N - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):
        super(FeedforwardNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]  
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, output_size))  
        self.layers = nn.ModuleList(layers)  
        self.activate = nn.ReLU()
        
    def forward(self, x):
        for layer in self.layers[:-1]:  
            x = self.activate(layer(x))  
        x = self.layers[-1](x)  
        return x

model = FeedforwardNN(input_size=1, hidden_size=width, output_size=1, hidden_layers=depth)
# print(model)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()  
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))
    
model.eval()

# 准备数据以绘图
x_test = []
y_test_actual = []
y_test_pred = []
test_losses = []

with torch.no_grad():  # 关闭梯度计算
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_losses.append(loss.item())
        x_test.extend(inputs.cpu().numpy())
        y_test_actual.extend(targets.cpu().numpy())
        y_test_pred.extend(outputs.cpu().numpy())

# 转换为numpy数组以便绘图
x_test = np.array(x_test).flatten()
y_test_actual = np.array(y_test_actual).flatten()
y_test_pred = np.array(y_test_pred).flatten()


# 绘制验证集原始样本点和预测点
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test_actual, color='red', label='Actual', alpha=0.5)
plt.scatter(x_test, y_test_pred, color='blue', label='Predicted', alpha=0.5)
title = f'N={N}, Relu, width={width}, learning rate={learning_rate}, depth={depth}'
plt.title(title)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
# file_name = f'lr_{learning_rate}.png'
plt.savefig('Relu.png')
plt.close() 

# 绘制训练过程中的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_loss.png')
plt.close()
