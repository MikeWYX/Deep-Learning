import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from models import GCN_LP
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

device = "cuda:4" if torch.cuda.is_available() else "cpu"
layer_num = 4
hidden_channels = 256
out_channels = 32
dropout = 0.5
learning_rate = 1e-3
decay = 5e-3
epochs = 100
pair_norm = True
drop_prob = 0.0

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])
dataset = Planetoid(root='//data/wuyux', name='Cora', transform=transform)
train_data, val_data, test_data = dataset[0]

def negative_sample():
    # 从训练集中采样与正边相同数量的负边
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    # print(neg_edge_index.size(1))   # 3642条负边，即每次采样与训练集中正边数量一致的负边
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index

model = GCN_LP(dataset.num_features, hidden_channels, out_channels, dropout, layer_num, pair_norm, drop_prob)
model.to(device)
# print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
def lambda_lr(epoch):
    if epoch >= 75:
        return 0.01 
    elif epoch >= 10:
        return 0.1  
    return 1.0   
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

def train():
    model.train()
    optimizer.zero_grad()
    edge_label, edge_label_index = negative_sample()
    out = model(train_data.x, train_data.edge_index, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    out_sigmoid = out.sigmoid()
    train_auc = roc_auc_score(edge_label.cpu().numpy(), out_sigmoid.detach().cpu().numpy())    
    return loss.item(), train_auc

def validate():
    model.eval()
    with torch.no_grad():
        z = model.encode(val_data.x, val_data.edge_index)
        out = model.decode(z, val_data.edge_label_index).view(-1)
        val_loss = criterion(out, val_data.edge_label).item()
        out_sigmoid = out.sigmoid()
    return val_loss, roc_auc_score(val_data.edge_label.cpu().numpy(), out_sigmoid.cpu().numpy())

def test():
    model.eval()
    with torch.no_grad():
        z = model.encode(test_data.x, test_data.edge_index)
        out = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(test_data.edge_label.cpu().numpy(), out.cpu().numpy())

train_losses = []
val_losses = []
best_val_auc = 0
best_model_path = "dl/pytorch/exp3/param/citeseer_link_prediction.pth"
for epoch in range(epochs):
    train_loss, train_auc = train()
    val_loss, val_auc = validate()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:3d}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    scheduler.step()

model.load_state_dict(torch.load(best_model_path))
test_auc = test()
print(f"Best Validation AUC: {best_val_auc:.4f}  at epoch {best_epoch}")
print(f"Test AUC: {test_auc:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')
plt.grid(True)
plt.savefig("dl/pytorch/exp3/param/citeseer_link_prediction_loss.png")  