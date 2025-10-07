import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
from timm.models.layers import trunc_normal_
from dataload import load_selfVR_data_cross_subject,load_HGD_data_cross_subject,load_bciciv2a_data_cross_subject,load_HGD_single_subject,load_selfVR_single_subject,load_bciciv2a_data_single_subject


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        super().__init__()
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x ** 2, dim=self.dim), 1e-4, 1e4))

class InterFre(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = sum(x)
        return F.gelu(out)

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class Stem(nn.Module):
    def __init__(self, in_planes, out_planes=64, kernel_size=63, patch_size=125, radix=2):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.patch_size = patch_size

        self.sconv = Conv(nn.Conv1d(in_planes, self.mid_planes, 1, bias=False, groups=radix),
                          bn=nn.BatchNorm1d(self.mid_planes))

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(out_planes, out_planes, kernel_size, 1,
                                             groups=out_planes, padding=kernel_size // 2, bias=False),
                                   bn=nn.BatchNorm1d(out_planes)))
            kernel_size //= 2

        self.interFre = InterFre()
        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]
        out = self.interFre(out)
        out = out.reshape(N, self.out_planes, T // self.patch_size, self.patch_size)
        out = self.power(out)
        out = self.dp(out)
        return out

class IFNet(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, radix, patch_size, time_points, num_classes):
        super().__init__()
        self.stem = Stem(in_planes * radix, out_planes, kernel_size, patch_size, radix)
        self.fc = nn.Sequential(
            LinearWithConstraint(out_planes * (time_points // patch_size), num_classes)
        )
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):

            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv1d):

            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm1d):

            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.stem(x)
        out = self.fc(out.flatten(1))
        return out


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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = '/root/autodl-tmp/EEG/KGAT-Mamba/data/HGD_npy'

radix = 1
patch_size = 100
kernel_size = 63
num_classes = 4  #4、4、2
out_planes = 64
time_points = 1000   # 1000、1000、768
in_channels = 44

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

    model = IFNet(
        in_planes=in_channels,
        out_planes=out_planes,
        kernel_size=kernel_size,
        radix=radix,
        patch_size=patch_size,
        time_points=time_points,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_kappa, best_epoch = 0.0, 0.0, -1
    last_acc, last_kappa = 0.0, 0.0

    for epoch in range(201):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        acc, preds, labels = evaluate_model(model, test_loader, device)
        kappa = cohen_kappa_score(labels, preds)

        if acc > best_acc:
            best_acc, best_kappa, best_epoch = acc, kappa, epoch + 1
        last_acc, last_kappa = acc, kappa

        print(f"Subject {subject_id} Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Acc={acc:.4f}, Kappa={kappa:.4f}")

    all_results.append({
        'subject': subject_id,
        'best_acc': best_acc,
        'best_kappa': best_kappa,
        'best_epoch': best_epoch,
        'last_acc': last_acc,
        'last_kappa': last_kappa
    })


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