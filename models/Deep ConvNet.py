import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
from dataload import load_selfVR_data_cross_subject,load_HGD_data_cross_subject,load_bciciv2a_data_cross_subject,load_HGD_single_subject,load_selfVR_single_subject,load_bciciv2a_data_single_subject
# ==================== Deep ConvNet ====================
class DeepConvNet(nn.Module):
    def __init__(self, num_channels, seq_length, num_classes, dropout_rate=0.5):
        super().__init__()
        p_feat = min(float(dropout_rate), 0.30)
        p_head = 0.20

        self.conv_block = nn.Sequential(
            # Block 1: temporal conv -> spatial conv
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=1, bias=False),
            nn.Conv2d(25, 25, kernel_size=(num_channels, 1), stride=1, bias=False),
            nn.InstanceNorm2d(25, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p_feat),

            # Block 2
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=1, bias=False),
            nn.InstanceNorm2d(50, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p_feat),

            # Block 3
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=1, bias=False),
            nn.InstanceNorm2d(100, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p_feat),

            # Block 4
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=1, bias=False),
            nn.InstanceNorm2d(200, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p_feat),
        )

        self.feature_dim = self._compute_feature_dim(num_channels, seq_length)


        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p_head),
            nn.Linear(self.feature_dim, num_classes)
        )

        self.apply(self._init_weights)

    def _compute_feature_dim(self, num_channels, seq_length):
        x_dummy = torch.zeros(1, 1, num_channels, seq_length)
        with torch.no_grad():
            x_dummy = self.conv_block(x_dummy)
        return x_dummy.view(1, -1).shape[1]

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)    # γ=1
            if m.bias is not None:
                nn.init.zeros_(m.bias)     # β=0
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)            # [B, 1, C, T]
        x = self.conv_block(x)        # [B, 200, 1, T']
        x = x.view(x.size(0), -1)     # flatten
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


def run_experiment(X_train, y_train, X_test, y_test,
                   num_channels, seq_length, num_classes,
                   lr=1e-3, epochs=50, batch_size=64, dropout=0.5):

    train_set = EEGDataset(X_train, y_train)
    test_set = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = DeepConvNet(num_channels, seq_length, num_classes, dropout).to(device)
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = '/home/fafu/lrq/EEG/KGAT-Mamba/data/HGD_npy'   # 2a/HGD_npy/VR-MI
from sklearn.metrics import cohen_kappa_score, accuracy_score

all_results = []

for subject_id in range(1,21):
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

    # model = DeepConvNet(num_channels=22, seq_length=1000, num_classes=4, dropout_rate=0.5).to(device)
    model = DeepConvNet(num_channels=44, seq_length=1000, num_classes=4, dropout_rate=0.5).to(device)
    # model = DeepConvNet(num_channels=32, seq_length=768, num_classes=2, dropout_rate=0.5).to(device)

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
