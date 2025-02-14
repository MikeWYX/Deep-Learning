import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from models import GCN
import matplotlib.pyplot as plt

device = "cuda:2" if torch.cuda.is_available() else "cpu"
layer_num = 2
hidden_channels = 1024
dropout = 0.1
learning_rate = 5e-4
decay = 5e-3
epochs = 100
pair_norm = True
drop_prob = 0.0

dataset = Planetoid(root='/data/wuyux', name='citeseer', transform=NormalizeFeatures())
data = dataset[0].to(device)

def edge_index_to_adj(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=edge_index.device)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    return adj_matrix

adj_matrix = edge_index_to_adj(data.edge_index, data.num_nodes)

def drop_edge(adj_matrix, drop_prob):
    mask = torch.rand(adj_matrix.size()) > drop_prob
    mask = mask.to(adj_matrix.device)
    adj_matrix_dropped = adj_matrix * mask
    return adj_matrix_dropped

model = GCN(dataset.num_features, hidden_channels, dataset.num_classes, dropout, layer_num, pair_norm)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
def lambda_lr(epoch):
    if epoch >= 200:
        return 0.25 
    elif epoch >= 100:
        return 0.5  
    return 1.0   
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, adj_matrix_dropped)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    correct = out.argmax(dim=1)[data.train_mask] == data.y[data.train_mask]
    train_acc = correct.sum().item() / data.train_mask.sum().item()
    loss.backward()
    optimizer.step()
    return loss.item(), train_acc

def validate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, adj_matrix)
        pred = out.argmax(dim=1)
        correct = pred[data.val_mask] == data.y[data.val_mask]
        val_acc = correct.sum().item() / data.val_mask.sum().item()
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return val_loss.item(), val_acc

def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, adj_matrix)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = correct.sum().item() / data.test_mask.sum().item()
    return test_acc

train_losses = []
val_losses = []
best_val_acc = 0
best_model_path = "dl/pytorch/exp3/param/citeseer_node_classification.pth"
for epoch in range(epochs):
    adj_matrix_dropped = drop_edge(adj_matrix, drop_prob)
    train_loss, train_acc = train()
    val_loss, val_acc = validate()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
    val_losses.append(val_loss)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}, Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, train_acc {train_acc:.4f}, val_acc {val_acc:.4f}")
    scheduler.step()

model.load_state_dict(torch.load(best_model_path))
test_acc = test()
print(f"Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
print(f"Test Accuracy: {test_acc:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')
plt.grid(True)
plt.savefig("dl/pytorch/exp3/param/citeseer_node_classification_loss.png")   