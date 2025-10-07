import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
from dataload import load_selfVR_data_cross_subject,load_HGD_data_cross_subject,load_bciciv2a_data_cross_subject,load_HGD_single_subject,load_selfVR_single_subject,load_bciciv2a_data_single_subject

# ==================== Shallow ConvNet ====================
class ShallowConvNet(nn.Module):
    def __init__(self, in_channels=44, input_time=1000, num_classes=4):
        super().__init__()


        self.conv_time = nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, bias=False)
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(in_channels, 1), stride=1, bias=False)


        self.norm = nn.InstanceNorm2d(40, affine=True, eps=1e-5)

        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))

        self.dropout = nn.Dropout(0.25)

        self.flat_dim = self._infer_flat_dim(in_channels, input_time)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.flat_dim, eps=1e-5),
            nn.Dropout(0.20),
            nn.Linear(self.flat_dim, num_classes)
        )

        self.apply(self._init_weights)

    @torch.no_grad()
    def _infer_flat_dim(self, C, T):
        x = torch.zeros(1, 1, C, T)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.norm(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return x.view(1, -1).shape[1]

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)   # γ=1
            if m.bias is not None:
                nn.init.zeros_(m.bias)    # β=0
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, C, T]
        x = x.unsqueeze(1)                 # [B, 1, C, T]
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.norm(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, 1e-6, 1e6))
        x = self.dropout(x)
        x = x.flatten(1)
        return self.classifier(x)



class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = '/home/fafu/lrq/EEG/KGAT-Mamba/data/HGD_npy'

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

    # model = ShallowConvNet(in_channels=22, input_time=1000, num_classes=4).to(device)
    model = ShallowConvNet(in_channels=44, input_time=1000, num_classes=4).to(device)
    # model = ShallowConvNet(in_channels=32, input_time=768, num_classes=2).to(device)

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