import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
from dataload import load_selfVR_data_cross_subject,load_HGD_data_cross_subject,load_bciciv2a_data_cross_subject,load_HGD_single_subject,load_selfVR_single_subject,load_bciciv2a_data_single_subject
# ==================== FBCSPNet ====================
class FBCSPNet(nn.Module):
    def __init__(self, num_channels, seq_length, num_classes, num_filters=8):
        super(FBCSPNet, self).__init__()

        self.filter_banks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_filters, kernel_size=(1, 15 + i * 10), padding='same', bias=False),
                nn.BatchNorm2d(num_filters)
            ) for i in range(4)
        ])

        self.spatial_filters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_filters, num_filters * 2, kernel_size=(num_channels, 1), bias=False),
                nn.BatchNorm2d(num_filters * 2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
                nn.Dropout(0.5)
            ) for _ in range(4)
        ])

        self.temporal_layer = nn.Sequential(
            nn.Conv2d(num_filters * 8, 64, kernel_size=(1, 7), padding='same'),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )

        self.out_size = seq_length // 8

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 32))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)  # [batch, 1, channels, time]

        filter_outputs = []
        for i, filter_bank in enumerate(self.filter_banks):
            fb_out = filter_bank(x)
            sp_out = self.spatial_filters[i](fb_out)
            filter_outputs.append(sp_out)

        x = torch.cat(filter_outputs, dim=1)  # [batch, filters*8, 1, time/4]

        x = self.temporal_layer(x)  # [batch, 64, 1, time/8]

        att = self.attention(x)
        x = x * att

        x = self.global_pool(x)  # [batch, 64, 1, 32]

        x = x.view(batch_size, -1)
        x = self.classifier(x)

        return x


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()  # 加一个 squeeze 防止多维标签

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


def run_experiment(X_train, y_train, X_test, y_test,
                   num_channels, seq_length, num_classes,
                   lr=1e-3, epochs=50, batch_size=64, dropout=0.5):

    train_set = EEGDataset(X_train, y_train)
    test_set = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = FBCSPNet(num_channels, seq_length, num_classes, dropout).to(device)
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = '/root/autodl-tmp/EEG/KGAT-Mamba/data/HGD_npy'
from sklearn.metrics import cohen_kappa_score, accuracy_score

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

    # model = FBCSPNet(num_channels=22, seq_length=1000, num_classes=4).to(device).to(device)
    model = FBCSPNet(num_channels=44, seq_length=1000, num_classes=4).to(device).to(device)
    # model = FBCSPNet(num_channels=32, seq_length=768, num_classes=2).to(device)

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
