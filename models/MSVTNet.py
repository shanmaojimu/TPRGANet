import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
from einops.layers.torch import Rearrange
from dataload import load_selfVR_data_cross_subject,load_HGD_data_cross_subject,load_bciciv2a_data_cross_subject,load_HGD_single_subject,load_selfVR_single_subject,load_bciciv2a_data_single_subject



# ==================== MSVTNet Model ====================
class TSConv(nn.Sequential):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc),
            nn.Conv2d(F * D, F * D, (1, C2), padding='same', groups=F * D, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc)
        )


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        x += self.pe
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            seq_len,
            d_model,
            nhead,
            ff_ratio,
            Pt=0.5,
            num_layers=4,
    ) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = PositionalEncoding(seq_len + 1, d_model)

        dim_ff = d_model * ff_ratio
        self.dropout = nn.Dropout(Pt)
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, Pt, batch_first=True, norm_first=True
        ), num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, x):
        b = x.shape[0]
        x = torch.cat((self.cls_embedding.expand(b, -1, -1), x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return self.trans(x)[:, 0]


class ClsHead(nn.Sequential):
    def __init__(self, linear_in, cls):
        super().__init__(
            nn.Flatten(),
            nn.Linear(linear_in, cls),
            nn.LogSoftmax(dim=1)
        )


class MSVTNet(nn.Module):
    def __init__(
            self,
            nCh=22,
            nTime=1000,
            cls=4,
            F=[9, 9, 9, 9],
            C1=[15, 31, 63, 125],
            C2=15,
            D=2,
            P1=8,
            P2=7,
            Pc=0.3,
            nhead=8,
            ff_ratio=1,
            Pt=0.5,
            layers=2,
            b_preds=True,
    ) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            nn.Sequential(
                TSConv(nCh, F[b], C1[b], C2, D, P1, P2, Pc),
                Rearrange('b d 1 t -> b t d')
            )
            for b in range(len(F))
        ])
        branch_linear_in = self._forward_flatten(cat=False)
        self.branch_head = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], cls)
            for b in range(len(F))
        ])

        seq_len, d_model = self._forward_mstsconv().shape[1:3]
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1]
        self.last_head = ClsHead(linear_in, cls)

    def _forward_mstsconv(self, cat=True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat=True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: [B, 1, C, T]
        x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head)]
        x = torch.cat(x, dim=2)
        x = self.transformer(x)
        x = self.last_head(x)
        if self.b_preds:
            return x, bx
        else:
            return x


class JointCrossEntropyLoss(nn.Module):
    def __init__(self, lamd: float = 0.6) -> None:
        super().__init__()
        self.lamd = lamd

    def forward(self, out, label):
        end_out = out[0]
        branch_out = out[1]
        end_loss = F.nll_loss(end_out, label)
        branch_loss = [F.nll_loss(out, label).unsqueeze(0) for out in branch_out]
        branch_loss = torch.cat(branch_loss)
        loss = self.lamd * end_loss + (1 - self.lamd) * torch.sum(branch_loss)
        return loss


# ==================== EEG Dataset ====================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== Training Function ====================
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


# ==================== Evaluation Function ====================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            # (main_output, branch_outputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels


# ==================== Experiment Main ====================
def run_experiment(X_train, y_train, X_test, y_test,
                   num_channels, seq_length, num_classes,
                   lr=1e-3, epochs=50, batch_size=64, dropout=0.5):
    train_set = EEGDataset(X_train, y_train)
    test_set = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = MSVTNet(nCh=num_channels, nTime=seq_length, cls=num_classes, Pc=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = JointCrossEntropyLoss(lamd=0.6)

    train_losses = []
    test_accs = []
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc, _, _ = evaluate_model(model, test_loader, device)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = '/root/autodl-tmp/EEG/KGAT-Mamba/data/HGD_npy'

all_results = []

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

    # model = MSVTNet(nCh=22, nTime=1000, cls=4, Pc=0.5).to(device)
    model = MSVTNet(nCh=44, nTime=1000, cls=4, Pc=0.5).to(device)
    # model = MSVTNet(nCh=32, nTime=768, cls=2, Pc=0.5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = JointCrossEntropyLoss(lamd=0.6)

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
        print(
            f"Subject {subject_id} Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Test Acc={acc:.4f}, Kappa={kappa:.4f}")

    all_results.append({
        'subject': subject_id,
        'best_acc': best_acc,
        'best_kappa': best_kappa,
        'best_epoch': best_epoch,
        'last_acc': last_acc,
        'last_kappa': last_kappa
    })

# Summary
best_accs = [r['best_acc'] for r in all_results]
best_kappas = [r['best_kappa'] for r in all_results]
last_accs = [r['last_acc'] for r in all_results]
last_kappas = [r['last_kappa'] for r in all_results]

print("\nSummary for all subjects:")
for r in all_results:
    print(
        f"Subject {r['subject']}: Best Acc={r['best_acc']:.4f} (Epoch {r['best_epoch']}), Best Kappa={r['best_kappa']:.4f}, Last Acc={r['last_acc']:.4f}, Last Kappa={r['last_kappa']:.4f}")

print(f"\nAverage Best Accuracy: {np.mean(best_accs):.4f}")
print(f"Average Best Kappa: {np.mean(best_kappas):.4f}")
print(f"Average Last Accuracy: {np.mean(last_accs):.4f}")
print(f"Average Last Kappa: {np.mean(last_kappas):.4f}")