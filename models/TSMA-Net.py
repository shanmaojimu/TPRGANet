import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
import math
from einops import rearrange
from dataload import load_selfVR_data_cross_subject,load_HGD_data_cross_subject,load_bciciv2a_data_cross_subject,load_HGD_single_subject,load_selfVR_single_subject,load_bciciv2a_data_single_subject

# Configuration class for TMSANet parameters
class Config:
    pool_size = 50
    pool_stride = 15
    num_heads = 4
    fc_ratio = 2
    depth = 1

config = Config()

# Multi-scale 1D Convolution Module
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p) for k, p in zip(kernel_sizes, padding)
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        out = self.bn(out)
        out = self.dropout(out)
        return out

# Multi-Headed Attention Module
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head
        kernel_sizes = [3, 5]
        padding = [1, 2]
        self.multi_scale_conv_k = MultiScaleConv1d(d_model, d_model, kernel_sizes, padding)
        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k_local = nn.Linear(d_model * len(kernel_sizes), n_head * self.d_k)
        self.w_k_global = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        bsz = query.size(0)
        key_local = key.transpose(1, 2)
        key_local = self.multi_scale_conv_k(key_local).transpose(1, 2)
        q = self.w_q(query).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        k_local = self.w_k_local(key_local).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        k_global = self.w_k_global(key).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(bsz, -1, self.n_head, self.d_v).transpose(1, 2)
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_local = F.softmax(scores_local, dim=-1)
        attn_local = self.dropout(attn_local)
        x_local = torch.matmul(attn_local, v)
        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_global = F.softmax(scores_global, dim=-1)
        attn_global = self.dropout(attn_global)
        x_global = torch.matmul(attn_global, v)
        x = x_local + x_global
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.n_head * self.d_v)
        return self.w_o(x)

# Feed-Forward Neural Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x

# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)
        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

# Feature Extraction Module
class ExtractFeature(nn.Module):
    def __init__(self, num_channels, num_samples, embed_dim, pool_size, pool_stride):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(1, embed_dim, (1, 31), padding=(0, 15))
        self.temp_conv2 = nn.Conv2d(1, embed_dim, (1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.spatial_conv1 = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.glu = nn.GELU()
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x = x1 + x2
        x = self.bn1(x)
        x = self.spatial_conv1(x)
        x = self.glu(x)
        x = self.bn2(x)
        x = x.squeeze(dim=2)
        x = self.avg_pool(x)
        return x

# Transformer Module
class TransformerModule(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop):
        super().__init__()
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)
        ])

    def forward(self, x):
        x = rearrange(x, 'b d n -> b n d')
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=2)
        return x

# Classification Module
class ClassifyModule(nn.Module):
    def __init__(self, embed_dim, temp_embedding_dim, num_classes):
        super().__init__()
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.classify(x)
        return out

# Complete TMSA-Net Model
class TMSANet(nn.Module):
    def __init__(self, in_planes, radix, time_points, num_classes, embed_dim=19, pool_size=config.pool_size,
                 pool_stride=config.pool_stride, num_heads=config.num_heads, fc_ratio=config.fc_ratio, depth=config.depth, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.in_planes = in_planes * radix
        self.extract_feature = ExtractFeature(self.in_planes, time_points, embed_dim, pool_size, pool_stride)
        temp_embedding_dim = (time_points - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout()
        self.transformer_module = TransformerModule(embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop)
        self.classify_module = ClassifyModule(embed_dim, temp_embedding_dim, num_classes)

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.dropout(x)
        x = self.transformer_module(x)
        out = self.classify_module(x)
        return out

# EEG Dataset
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Training Function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

# Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels

# Main Experiment Function
def run_experiment(X_train, y_train, X_test, y_test,
                   in_planes, seq_length, num_classes,
                   lr=1e-3, epochs=50, batch_size=64, dropout=0.5):

    train_set = EEGDataset(X_train, y_train)
    test_set = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = TMSANet(in_planes=in_planes, radix=1, time_points=seq_length, num_classes=num_classes, embed_dim=19, attn_drop=dropout, fc_drop=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_accs = []
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc, _, _ = evaluate_model(model, test_loader, device)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")
    return model

# Main Script
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = '/root/autodl-tmp/EEG/KGAT-Mamba/data/HGD_npy'

all_results = []  # Store results for all subjects

for subject_id in range(1, 15):
    print(f"Training Subject {subject_id} ...")
    # train_X, train_y, test_X, test_y = load_bciciv2a_data_single_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_bciciv2a_data_cross_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_HGD_single_subject(data_path, subject_id)
    train_X, train_y, test_X, test_y = load_HGD_data_cross_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_selfVR_single_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_selfVR_data_cross_subject(data_path, subject_id)

    train_set = EEGDataset(train_X, train_y)
    test_set = EEGDataset(test_X, test_y)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # model = TMSANet(in_planes=22, radix=1, time_points=1000, num_classes=4, embed_dim=19, attn_drop=0.5, fc_drop=0.5).to(device)
    model = TMSANet(in_planes=44, radix=1, time_points=1000, num_classes=4, embed_dim=19, attn_drop=0.5, fc_drop=0.5).to(device)
    # model = TMSANet(in_planes=32, radix=1, time_points=768, num_classes=4, embed_dim=19, attn_drop=0.5, fc_drop=0.5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_kappa = 0.0
    best_epoch = -1
    last_acc = 0.0
    last_kappa = 0.0

    for epoch in range(201):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        acc, preds, labels = evaluate_model(model, test_loader, device)
        kappa = cohen_kappa_score(labels, preds)

        if acc > best_acc:
            best_acc = acc
            best_kappa = kappa
            best_epoch = epoch + 1

        last_acc = acc
        last_kappa = kappa
        print(f"Subject {subject_id} Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Acc={acc:.4f}, Kappa={kappa:.4f}")

    all_results.append({
        'subject': subject_id,
        'best_acc': best_acc,
        'best_kappa': best_kappa,
        'best_epoch': best_epoch,
        'last_acc': last_acc,
        'last_kappa': last_kappa
    })

# Summary Statistics
best_accs = [r['best_acc'] for r in all_results]
best_kappas = [r['best_kappa'] for r in all_results]
last_accs = [r['last_acc'] for r in all_results]
last_kappas = [r['last_kappa'] for r in all_results]

print("\nSummary for all subjects:")
for r in all_results:
    print(f"Subject {r['subject']}: Best Acc={r['best_acc']:.4f} (Epoch {r['best_epoch']}), Best Kappa={r['best_kappa']:.4f}, Last Acc={r['last_acc']:.4f}, Last Kappa={r['last_kappa']:.4f}")

print(f"\nAverage Best Accuracy: {np.mean(best_accs):.4f}")
print(f"Average Best Kappa: {np.mean(best_kappas):.4f}")
print(f"Average Last Accuracy: {np.mean(last_accs):.4f}")
print(f"Average Last Kappa: {np.mean(last_kappas):.4f}")