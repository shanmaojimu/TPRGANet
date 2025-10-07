import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from einops import rearrange
from dataload import load_selfVR_data_cross_subject,load_HGD_data_cross_subject,load_bciciv2a_data_cross_subject,load_HGD_single_subject,load_selfVR_single_subject,load_bciciv2a_data_single_subject


def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**0.5
    attn = nn.functional.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn

class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []
        for i in range(out_shape):
            idx = i * self.stride
            segment = x[:, :, idx:idx+self.kernel_size]
            output = torch.log(torch.clamp(segment.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)
        return torch.cat(out, dim=-1)

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), 'b n (h d) -> b h n d', h=self.n_head)
        k = rearrange(self.w_k(key),  'b n (h d) -> b h n d', h=self.n_head)
        v = rearrange(self.w_v(value),'b n (h d) -> b h n d', h=self.n_head)
        out, _ = attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.dropout(self.w_o(out))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.mha = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.ffn = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TransNet(nn.Module):
    def __init__(self, num_classes=2, num_samples=768, num_channels=32,
                 embed_dim=32, pool_size=50, pool_stride=15, num_heads=8,
                 fc_ratio=4, depth=4, attn_drop=0.2, fc_drop=0.2):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(1, embed_dim // 4, (1, 15), padding=(0, 7), bias=False)
        self.temp_conv2 = nn.Conv2d(1, embed_dim // 4, (1, 25), padding=(0, 12), bias=False)
        self.temp_conv3 = nn.Conv2d(1, embed_dim // 4, (1, 51), padding=(0, 25), bias=False)
        self.temp_conv4 = nn.Conv2d(1, embed_dim // 4, (1, 65), padding=(0, 32), bias=False)
        self.bn1 = nn.InstanceNorm2d(embed_dim, affine=True, eps=1e-5)

        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), bias=False)
        self.bn2 = nn.InstanceNorm2d(embed_dim, affine=True, eps=1e-5)
        self.elu = nn.ELU()

        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        self.temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout(0.25)

        self.transformers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)
        ])

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(self.temp_embedding_dim, self.temp_embedding_dim, (2, 1), bias=False),
            nn.InstanceNorm2d(self.temp_embedding_dim, affine=True, eps=1e-5),
            nn.ELU()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * self.temp_embedding_dim, eps=1e-5),
            nn.Dropout(0.20),
            nn.Linear(embed_dim * self.temp_embedding_dim, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu' if isinstance(m, nn.Conv2d) else 'linear')
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None: nn.init.ones_(m.weight)
            if m.bias   is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = torch.cat([self.temp_conv1(x), self.temp_conv2(x),
                       self.temp_conv3(x), self.temp_conv4(x)], dim=1)
        x = self.bn1(x)
        x = self.elu(self.bn2(self.spatial_conv(x)))
        x = x.squeeze(2)  # [B, D, T]

        x1 = self.dropout(self.avg_pool(x))
        x2 = self.dropout(self.var_pool(x))
        x1 = rearrange(x1, 'b d t -> b t d')  # [B, T', D]
        x2 = rearrange(x2, 'b d t -> b t d')

        for block in self.transformers:
            x1 = block(x1)
            x2 = block(x2)

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2)  # [B, T', 2, D]
        x = self.conv_encoder(x)                                  # [B, T', 1, D]
        x = x.reshape(x.size(0), -1)                              # [B, T'*D]
        return self.classifier(x)


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.numpy())
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    return acc, kappa, preds, labels


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = '/home/fafu/lrq/EEG/KGAT-Mamba/data/VR-MI'

all_results = []

for subject_id in range(1, 21):
    print(f"\n--- Training Subject {subject_id} ---")
    # train_X, train_y, test_X, test_y = load_bciciv2a_data_single_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_bciciv2a_data_cross_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_HGD_single_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_HGD_data_cross_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_selfVR_single_subject(data_path, subject_id)
    train_X, train_y, test_X, test_y = load_selfVR_data_cross_subject(data_path, subject_id)

    train_loader = DataLoader(EEGDataset(train_X, train_y), batch_size=64, shuffle=True)
    test_loader = DataLoader(EEGDataset(test_X, test_y), batch_size=64, shuffle=False)

    model = TransNet(
        num_classes=2, num_samples=768, num_channels=32,
        embed_dim=32, pool_size=50, pool_stride=15,
        num_heads=8, fc_ratio=4, depth=4,
        attn_drop=0.5, fc_drop=0.5
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_kappa = 0
    last_acc = 0
    last_kappa = 0
    best_epoch = -1

    for epoch in range(201):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        acc, kappa, _, _ = evaluate_model(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_kappa = kappa
            best_epoch = epoch + 1
        last_acc = acc
        last_kappa = kappa
        print(f"Subject {subject_id} | Epoch {epoch+1} | Loss={train_loss:.4f} | Acc={acc:.4f} | Kappa={kappa:.4f}")

    all_results.append({
        'subject': subject_id,
        'best_acc': best_acc,
        'best_kappa': best_kappa,
        'best_epoch': best_epoch,
        'last_acc': last_acc,
        'last_kappa': last_kappa
    })


print("\n=== Final Summary ===")
for r in all_results:
    print(f"Subject {r['subject']}: Best Acc={r['best_acc']:.4f} (Epoch {r['best_epoch']}), "
          f"Best Kappa={r['best_kappa']:.4f}, Last Acc={r['last_acc']:.4f}, Last Kappa={r['last_kappa']:.4f}")

print(f"\nAverage Best Accuracy: {np.mean([r['best_acc'] for r in all_results]):.4f}")
print(f"Average Best Kappa: {np.mean([r['best_kappa'] for r in all_results]):.4f}")
print(f"Average Last Accuracy: {np.mean([r['last_acc'] for r in all_results]):.4f}")
print(f"Average Last Kappa: {np.mean([r['last_kappa'] for r in all_results]):.4f}")


#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# from sklearn.metrics import accuracy_score, cohen_kappa_score
# from einops import rearrange
# from dataload import (
#     load_selfVR_data_cross_subject,
#     load_HGD_data_cross_subject,
#     load_bciciv2a_data_cross_subject,
#     load_HGD_single_subject,
#     load_selfVR_single_subject,
#     load_bciciv2a_data_single_subject
# )
#
# # ==================== 实用函数 ====================
# def zscore_by_channel(train_X, test_X, eps=1e-6):
#     mean = train_X.mean(axis=(0, 2), keepdims=True)
#     std  = train_X.std(axis=(0, 2), keepdims=True) + eps
#     return (train_X - mean) / std, (test_X - mean) / std
#
# def class_weights_from_labels(y, num_classes, device):
#     y_t = torch.as_tensor(y).view(-1).long()
#     y_min = int(y_t.min().item())
#     if y_min != 0:
#         y_t = y_t - y_min
#     counts = torch.bincount(y_t, minlength=num_classes).float()
#     counts = torch.where(counts == 0, torch.ones_like(counts), counts)
#     inv = 1.0 / counts
#     w = inv / inv.sum() * num_classes
#     return w.to(device)
#
# def attention(query, key, value):
#     dim = query.size(-1)
#     scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**0.5
#     attn = nn.functional.softmax(scores, dim=-1)
#     out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
#     return out, attn
#
# class VarPoold(nn.Module):
#     def __init__(self, kernel_size, stride):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#     def forward(self, x):
#         t = x.shape[2]
#         out_shape = (t - self.kernel_size) // self.stride + 1
#         outs = []
#         for i in range(out_shape):
#             idx = i * self.stride
#             seg = x[:, :, idx:idx+self.kernel_size]
#             v = torch.log(torch.clamp(seg.var(dim=-1, keepdim=True), 1e-6, 1e6))
#             outs.append(v)
#         return torch.cat(outs, dim=-1)
#
# class MultiHeadedAttention(nn.Module):
#     def __init__(self, d_model, n_head, dropout):
#         super().__init__()
#         self.d_k = d_model // n_head
#         self.d_v = d_model // n_head
#         self.n_head = n_head
#         self.w_q = nn.Linear(d_model, n_head * self.d_k, bias=True)
#         self.w_k = nn.Linear(d_model, n_head * self.d_k, bias=True)
#         self.w_v = nn.Linear(d_model, n_head * self.d_v, bias=True)
#         self.w_o = nn.Linear(n_head * self.d_v, d_model, bias=True)
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, query, key, value):
#         q = rearrange(self.w_q(query), 'b n (h d) -> b h n d', h=self.n_head)
#         k = rearrange(self.w_k(key),  'b n (h d) -> b h n d', h=self.n_head)
#         v = rearrange(self.w_v(value),'b n (h d) -> b h n d', h=self.n_head)
#         out, _ = attention(q, k, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.dropout(self.w_o(out))
#
# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_hidden, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_model, d_hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_hidden, d_model),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x): return self.net(x)
#
# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.2, fc_drop=0.2):
#         super().__init__()
#         self.mha = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
#         self.ffn = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
#         self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
#         self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
#     def forward(self, x):
#         x = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#         return x
#
# class TransNet(nn.Module):
#     def __init__(self, num_classes=2, num_samples=768, num_channels=32,
#                  embed_dim=32, pool_size=50, pool_stride=15, num_heads=8,
#                  fc_ratio=4, depth=4, attn_drop=0.2, fc_drop=0.2):
#         super().__init__()
#         self.temp_conv1 = nn.Conv2d(1, embed_dim // 4, (1, 15), padding=(0, 7), bias=False)
#         self.temp_conv2 = nn.Conv2d(1, embed_dim // 4, (1, 25), padding=(0, 12), bias=False)
#         self.temp_conv3 = nn.Conv2d(1, embed_dim // 4, (1, 51), padding=(0, 25), bias=False)
#         self.temp_conv4 = nn.Conv2d(1, embed_dim // 4, (1, 65), padding=(0, 32), bias=False)
#         self.bn1 = nn.InstanceNorm2d(embed_dim, affine=True, eps=1e-5)
#
#         self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), bias=False)
#         self.bn2 = nn.InstanceNorm2d(embed_dim, affine=True, eps=1e-5)
#         self.elu = nn.ELU()
#
#         self.var_pool = VarPoold(pool_size, pool_stride)
#         self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
#         self.temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
#         self.dropout = nn.Dropout(0.25)
#
#         self.transformers = nn.ModuleList([
#             TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop)
#             for _ in range(depth)
#         ])
#
#         self.conv_encoder = nn.Sequential(
#             nn.Conv2d(self.temp_embedding_dim, self.temp_embedding_dim, (2, 1), bias=False),
#             nn.InstanceNorm2d(self.temp_embedding_dim, affine=True, eps=1e-5),
#             nn.ELU()
#         )
#
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(embed_dim * self.temp_embedding_dim, eps=1e-5),
#             nn.Dropout(0.20),
#             nn.Linear(embed_dim * self.temp_embedding_dim, num_classes)
#         )
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#         elif isinstance(m, nn.InstanceNorm2d):
#             if m.weight is not None: nn.init.ones_(m.weight)
#             if m.bias   is not None: nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
#             if m.bias is not None: nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # [B, 1, C, T]
#         x = torch.cat([self.temp_conv1(x), self.temp_conv2(x),
#                        self.temp_conv3(x), self.temp_conv4(x)], dim=1)
#         x = self.bn1(x)
#         x = self.elu(self.bn2(self.spatial_conv(x)))
#         x = x.squeeze(2)  # [B, D, T]
#
#         x1 = self.dropout(self.avg_pool(x))          # [B, D, T']
#         x2 = self.dropout(self.var_pool(x))          # [B, D, T']
#         x1 = rearrange(x1, 'b d t -> b t d')         # [B, T', D]
#         x2 = rearrange(x2, 'b d t -> b t d')
#
#         for block in self.transformers:
#             x1 = block(x1)
#             x2 = block(x2)
#
#         x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2)  # [B, T', 2, D]
#         x = self.conv_encoder(x)                                  # [B, T', 1, D]
#         x = x.reshape(x.size(0), -1)                              # [B, T'*D]
#         return self.classifier(x)
#
# class EEGDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long).squeeze()
#     def __len__(self): return len(self.X)
#     def __getitem__(self, idx): return self.X[idx], self.y[idx]
#
# def train_model(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     for x, y in dataloader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         out = model(x)
#         loss = criterion(out, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * x.size(0)
#     return total_loss / len(dataloader.dataset)
#
# @torch.no_grad()
# def evaluate_model(model, dataloader, device):
#     model.eval()
#     preds, labels = [], []
#     for x, y in dataloader:
#         x = x.to(device)
#         out = model(x)
#         pred = torch.argmax(out, dim=1).cpu().numpy()
#         preds.extend(pred)
#         labels.extend(y.numpy())
#     acc = accuracy_score(labels, preds)
#     kappa = cohen_kappa_score(labels, preds)
#     return acc, kappa, preds, labels
#
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# data_path = '/home/fafu/lrq/EEG/KGAT-Mamba/data/VR-MI'
#
# WARMUP_EPOCHS = 3            # 仅训练分类头的轮数
# TOTAL_EPOCHS  = 200          # 总轮数（含 warm-up）
# START_MAIN    = WARMUP_EPOCHS + 1
# MAIN_EPOCHS   = TOTAL_EPOCHS - WARMUP_EPOCHS  # = 197
#
# all_results = []
# for subject_id in range(1, 21):
#     print(f"\n--- Training Subject {subject_id} ---")
#     # VR 跨被试二分类
#     # train_X, train_y, test_X, test_y = load_bciciv2a_data_single_subject(data_path, subject_id)
#     # train_X, train_y, test_X, test_y = load_bciciv2a_data_cross_subject(data_path, subject_id)
#     # train_X, train_y, test_X, test_y = load_HGD_single_subject(data_path, subject_id)
#     # train_X, train_y, test_X, test_y = load_HGD_data_cross_subject(data_path, subject_id)
#     # train_X, train_y, test_X, test_y = load_selfVR_single_subject(data_path, subject_id)
#     train_X, train_y, test_X, test_y = load_selfVR_data_cross_subject(data_path, subject_id)
#
#     train_X, test_X = zscore_by_channel(train_X, test_X, eps=1e-6)
#
#     # DataLoader
#     train_loader = DataLoader(EEGDataset(train_X, train_y), batch_size=64, shuffle=True)
#     test_loader  = DataLoader(EEGDataset(test_X,  test_y),  batch_size=64, shuffle=False)
#
#     # 模型
#     model = TransNet(
#         num_classes=2, num_samples=768, num_channels=32,
#         embed_dim=32, pool_size=50, pool_stride=15,
#         num_heads=8, fc_ratio=4, depth=4,
#         attn_drop=0.2, fc_drop=0.2
#     ).to(device)
#
#     # 类权重
#     weights = class_weights_from_labels(train_y, num_classes=2, device=device)
#     criterion = nn.CrossEntropyLoss(weight=weights)
#
#     for p in model.parameters(): p.requires_grad = False
#     for p in model.classifier.parameters(): p.requires_grad = True
#     opt_head = optim.AdamW(model.classifier.parameters(), lr=3e-3, weight_decay=1e-4)
#
#     for epoch in range(1, WARMUP_EPOCHS + 1):
#         train_loss = train_model(model, train_loader, opt_head, criterion, device)
#         acc, kappa, _, _ = evaluate_model(model, test_loader, device)
#         print(f"[HeadWarmup] Epoch {epoch}/{WARMUP_EPOCHS} | Loss={train_loss:.4f} | Acc={acc:.4f} | Kappa={kappa:.4f}")
#
#     for p in model.parameters(): p.requires_grad = True
#     optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAIN_EPOCHS, eta_min=3e-5)
#
#     best_acc = best_kappa = 0.0
#     best_epoch = -1
#     last_acc = last_kappa = 0.0
#
#     for epoch in range(START_MAIN, START_MAIN + MAIN_EPOCHS):  # 4..200（共197轮）
#         train_loss = train_model(model, train_loader, optimizer, criterion, device)
#         acc, kappa, _, _ = evaluate_model(model, test_loader, device)
#
#         if acc > best_acc:
#             best_acc, best_kappa, best_epoch = acc, kappa, epoch
#         last_acc, last_kappa = acc, kappa
#
#         print(f"Subject {subject_id} | Epoch {epoch} | LR={scheduler.get_last_lr()[0]:.2e} "
#               f"| Loss={train_loss:.4f} | Acc={acc:.4f} | Kappa={kappa:.4f}")
#         scheduler.step()
#
#     all_results.append({
#         'subject': subject_id,
#         'best_acc': best_acc,
#         'best_kappa': best_kappa,
#         'best_epoch': best_epoch,
#         'last_acc': last_acc,
#         'last_kappa': last_kappa
#     })
#
# print("\n=== Final Summary ===")
# for r in all_results:
#     print(f"Subject {r['subject']}: Best Acc={r['best_acc']:.4f} (Epoch {r['best_epoch']}), "
#           f"Best Kappa={r['best_kappa']:.4f}, Last Acc={r['last_acc']:.4f}, Last Kappa={r['last_kappa']:.4f}")
#
# print(f"\nAverage Best Accuracy: {np.mean([r['best_acc'] for r in all_results]):.4f}")
# print(f"Average Best Kappa: {np.mean([r['best_kappa'] for r in all_results]):.4f}")
# print(f"Average Last Accuracy: {np.mean([r['last_acc'] for r in all_results]):.4f}")
# print(f"Average Last Kappa: {np.mean([r['last_kappa'] for r in all_results]):.4f}")
