import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
from typing import Optional, Tuple
from dataload import load_selfVR_data_cross_subject,load_HGD_data_cross_subject,load_bciciv2a_data_cross_subject,load_HGD_single_subject,load_selfVR_single_subject,load_bciciv2a_data_single_subject
from scipy.signal import butter, filtfilt

# ==================== Data Preprocessing for Frequency Bands ====================
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to EEG data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def preprocess_hgd_data(X, fs=250, num_bands=9):
    """Preprocess HGD data to extract 9 frequency bands."""
    # Input X: [batch, channels=44, time]
    freq_bands = [
        (4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
        (24, 28), (28, 32), (32, 36), (36, 40)
    ]  # 9 frequency bands
    X_bands = []
    for low, high in freq_bands[:num_bands]:
        X_filtered = bandpass_filter(X, low, high, fs)
        X_bands.append(X_filtered)
    # Stack to get [batch, num_bands=9, channels=44, time]
    X_bands = np.stack(X_bands, axis=1)
    return X_bands

# ==================== FBMSNet Support Classes ====================
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class VarLayer(nn.Module):
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def _get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def _same_pad_arg(input_size, kernel_size, stride, dilation):
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if _is_static_pad(kernel_size, **kwargs):
                padding = _get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic

def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        if isinstance(kernel_size, tuple):
            padding = (0, padding)
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

class MixedConv2d(nn.ModuleDict):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        # print(f"MixedConv2d input shape: {x.shape}, splits: {self.splits}")  # 调试打印
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x

# ==================== FBMSNet Model ====================
class FBMSNet(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))

    def __init__(self, nChan, nTime, nClass=4, temporalLayer='LogVarLayer', num_Feat=36, dilatability=8, dropoutP=0.5, *args, **kwargs):
        super(FBMSNet, self).__init__()
        self.strideFactor = 4
        self.mixConv2d = nn.Sequential(
            MixedConv2d(in_channels=9, out_channels=num_Feat, kernel_size=[(1, 15), (1, 31), (1, 63), (1, 125)],
                        stride=1, padding='', dilation=1, depthwise=False),
            nn.BatchNorm2d(num_Feat),
        )
        self.scb = self.SCB(in_chan=num_Feat, out_chan=num_Feat*dilatability, nChan=int(nChan))
        self.temporalLayer = globals()[temporalLayer](dim=3)
        size = self.get_size(nChan, nTime)
        self.fc = self.LastBlock(size[1], nClass)

    def forward(self, x):
        y = self.mixConv2d(x)
        x = self.scb(y)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        f = torch.flatten(x, start_dim=1)
        c = self.fc(f)
        return c, f

    def get_size(self, nChan, nTime):
        data = torch.ones((1, 9, nChan, nTime))
        x = self.mixConv2d(data)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()

# ==================== EEG Dataset ====================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        # Apply frequency band preprocessing
        X = preprocess_hgd_data(X, fs=250, num_bands=9)
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
        # print(f"DataLoader input shape: {x.shape}")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs, _ = model(x)  # FBMSNet returns (output, features)
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
            outputs, _ = model(x)  # FBMSNet returns (output, features)
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

    model = FBMSNet(nChan=num_channels, nTime=seq_length, nClass=num_classes, dropoutP=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

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
data_path = '/home/fafu/lrq/EEG/KGAT-Mamba/data/VR-MI'

all_results = []

for subject_id in range(1, 21):
    print(f"Training Subject {subject_id} ...")
    # train_X, train_y, test_X, test_y = load_bciciv2a_data_single_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_bciciv2a_data_cross_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_HGD_single_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_HGD_data_cross_subject(data_path, subject_id)
    # train_X, train_y, test_X, test_y = load_selfVR_single_subject(data_path, subject_id)
    train_X, train_y, test_X, test_y = load_selfVR_data_cross_subject(data_path, subject_id)

    train_set = EEGDataset(train_X, train_y)
    test_set = EEGDataset(test_X, test_y)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # model = FBMSNet(nChan=22, nTime=1000, nClass=4, dropoutP=0.5).to(device)
    # model = FBMSNet(nChan=44, nTime=1000, nClass=4, dropoutP=0.5).to(device)
    model = FBMSNet(nChan=32, nTime=768, nClass=2, dropoutP=0.5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

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

# Summary
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

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.metrics import accuracy_score, cohen_kappa_score
# import numpy as np
# from typing import Optional, Tuple
# from dataload import (
#     load_selfVR_data_cross_subject,
#     load_HGD_data_cross_subject,
#     load_bciciv2a_data_cross_subject,
#     load_HGD_single_subject,
#     load_selfVR_single_subject,
#     load_bciciv2a_data_single_subject
# )
# from scipy.signal import butter, filtfilt
#
# # ==================== Data Preprocessing for Frequency Bands ====================
# def bandpass_filter(data, lowcut, highcut, fs, order=5):
#     """Apply bandpass filter to EEG data along time axis (last dim)."""
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return filtfilt(b, a, data, axis=-1)
#
# def preprocess_hgd_data(X, fs=250, num_bands=9):
#     """X: [B, C, T] -> [B, 9, C, T] by 9 frequency bands."""
#     freq_bands = [
#         (4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
#         (24, 28), (28, 32), (32, 36), (36, 40)
#     ]
#     X_bands = []
#     for low, high in freq_bands[:num_bands]:
#         X_filtered = bandpass_filter(X, low, high, fs)
#         X_bands.append(X_filtered)
#     X_bands = np.stack(X_bands, axis=1)  # [B, 9, C, T]
#     return X_bands
#
# def band_zscore_stats(train_X_bands):
#     """Compute per-(band,channel) stats across time using training set."""
#     # train_X_bands: [B, 9, C, T]
#     mean = train_X_bands.mean(axis=(0, 3), keepdims=True)         # [1, 9, C, 1]
#     std  = train_X_bands.std(axis=(0, 3), keepdims=True) + 1e-6
#     return mean, std
#
# def apply_band_zscore(X_bands, mean, std):
#     return (X_bands - mean) / std
#
# # ==================== FBMSNet Support Classes ====================
# class Conv2dWithConstraint(nn.Conv2d):
#     def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
#         self.max_norm = max_norm
#         self.doWeightNorm = doWeightNorm
#         super().__init__(*args, **kwargs)
#     def forward(self, x):
#         if self.doWeightNorm:
#             self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
#         return super().forward(x)
#
# class LinearWithConstraint(nn.Linear):
#     def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
#         self.max_norm = max_norm
#         self.doWeightNorm = doWeightNorm
#         super().__init__(*args, **kwargs)
#     def forward(self, x):
#         if self.doWeightNorm:
#             self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
#         return super().forward(x)
#
# class VarLayer(nn.Module):
#     def __init__(self, dim): super().__init__(); self.dim = dim
#     def forward(self, x): return x.var(dim=self.dim, keepdim=True)
#
# class LogVarLayer(nn.Module):
#     def __init__(self, dim): super().__init__(); self.dim = dim
#     def forward(self, x): return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))
#
# class swish(nn.Module):
#     def forward(self, x): return x * torch.sigmoid(x)
#
# def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
#     return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0
#
# def _get_padding(kernel_size, stride=1, dilation=1, **_):
#     if isinstance(kernel_size, tuple):
#         kernel_size = max(kernel_size)
#     padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
#     return padding
#
# def _calc_same_pad(i: int, k: int, s: int, d: int):
#     return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)
#
# def conv2d_same(
#     x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
#     stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
#     dilation: Tuple[int, int] = (1, 1), groups: int = 1
# ):
#     ih, iw = x.size()[-2:]
#     kh, kw = weight.size()[-2:]
#     pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
#     pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
#     x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
#     return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
#
# class Conv2dSame(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
#     def forward(self, x): return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
# def get_padding_value(padding, kernel_size, **kwargs):
#     dynamic = False
#     if isinstance(padding, str):
#         padding = padding.lower()
#         if padding == 'same':
#             if _is_static_pad(kernel_size, **kwargs):
#                 padding = _get_padding(kernel_size, **kwargs)
#             else:
#                 padding = 0; dynamic = True
#         elif padding == 'valid':
#             padding = 0
#         else:
#             padding = _get_padding(kernel_size, **kwargs)
#     return padding, dynamic
#
# def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
#     padding = kwargs.pop('padding', '')
#     kwargs.setdefault('bias', False)
#     padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
#     if is_dynamic:
#         return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
#     else:
#         if isinstance(kernel_size, tuple):
#             padding = (0, padding)
#         return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
#
# def _split_channels(num_chan, num_groups):
#     split = [num_chan // num_groups for _ in range(num_groups)]
#     split[0] += num_chan - sum(split)
#     return split
#
# class MixedConv2d(nn.ModuleDict):
#     def __init__(self, in_channels, out_channels, kernel_size=3,
#                  stride=1, padding='', dilation=1, depthwise=False, **kwargs):
#         super().__init__()
#         kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
#         num_groups = len(kernel_size)
#         in_splits = _split_channels(in_channels, num_groups)
#         out_splits = _split_channels(out_channels, num_groups)
#         self.in_channels = sum(in_splits)
#         self.out_channels = sum(out_splits)
#         for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
#             conv_groups = out_ch if depthwise else 1
#             self.add_module(
#                 str(idx),
#                 create_conv2d_pad(in_ch, out_ch, k, stride=stride,
#                                   padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
#             )
#         self.splits = in_splits
#     def forward(self, x):
#         x_split = torch.split(x, self.splits, 1)
#         x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
#         return torch.cat(x_out, 1)
#
# # ==================== FBMSNet Model (Stabilized) ====================
# class FBMSNet(nn.Module):
#     def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
#         return nn.Sequential(
#             Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
#                                  max_norm=2, doWeightNorm=doWeightNorm, padding=0, bias=False),
#             nn.InstanceNorm2d(out_chan, affine=True, eps=1e-5),  # ← BN 改 IN
#             swish(),
#             nn.Dropout2d(0.10),
#         )
#
#     def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):

#         return nn.Sequential(
#             nn.LayerNorm(inF, eps=1e-5),
#             nn.Dropout(0.20),
#             LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
#         )
#
#     def __init__(self, nChan, nTime, nClass=4, temporalLayer='LogVarLayer',
#                  num_Feat=36, dilatability=8, dropoutP=0.5, *args, **kwargs):
#         super().__init__()
#         self.strideFactor = 4
#         self.mixConv2d = nn.Sequential(
#             MixedConv2d(in_channels=9, out_channels=num_Feat,
#                         kernel_size=[(1, 15), (1, 31), (1, 63), (1, 125)],
#                         stride=1, padding='', dilation=1, depthwise=False),
#             nn.InstanceNorm2d(num_Feat, affine=True, eps=1e-5),  # ← BN 改 IN
#         )
#         self.scb = self.SCB(in_chan=num_Feat, out_chan=num_Feat * dilatability, nChan=int(nChan))
#         self.temporalLayer = globals()[temporalLayer](dim=3)
#         size = self.get_size(nChan, nTime)
#         self.fc = self.LastBlock(size[1], nClass)
#
#     def forward(self, x):
#         # 输入: [B, 9, C, T]
#         y = self.mixConv2d(x)
#         x = self.scb(y)
#         x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
#         x = self.temporalLayer(x)
#         f = torch.flatten(x, start_dim=1)
#         logits = self.fc(f)
#         return logits, f
#
#     def get_size(self, nChan, nTime):
#         data = torch.ones((1, 9, nChan, nTime))
#         x = self.mixConv2d(data)
#         x = self.scb(x)
#         x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
#         x = self.temporalLayer(x)
#         x = torch.flatten(x, start_dim=1)
#         return x.size()
#
# # ==================== EEG Dataset with Band Standardization ====================
# class EEGDataset(Dataset):
#     def __init__(self, X, y, mean=None, std=None, fs=250, num_bands=9):
#         X_b = preprocess_hgd_data(X, fs=fs, num_bands=num_bands)     # [B, 9, C, T]
#         if (mean is not None) and (std is not None):
#             X_b = apply_band_zscore(X_b, mean, std)
#         self.X = torch.tensor(X_b, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long).squeeze()
#     def __len__(self): return self.X.shape[0]
#     def __getitem__(self, idx): return self.X[idx], self.y[idx]
#
# # ==================== Training / Evaluation ====================
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
# def train_model(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     for x, y in dataloader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         logits, _ = model(x)                 # logits
#         loss = criterion(logits, y)          # CrossEntropy
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
#         logits, _ = model(x)
#         pred = torch.argmax(logits, dim=1).cpu().numpy()
#         preds.extend(pred)
#         labels.extend(y.numpy())
#     acc = accuracy_score(labels, preds)
#     kappa = cohen_kappa_score(labels, preds)
#     return acc, kappa, preds, labels
#
# # ==================== Main Experiment ====================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_path = '/home/fafu/lrq/EEG/KGAT-Mamba/data/VR-MI'
#
# WARMUP_EPOCHS = 3            # 分类头热启动
# TOTAL_EPOCHS  = 200          # 总轮数=3+197
# START_MAIN    = WARMUP_EPOCHS + 1
# MAIN_EPOCHS   = TOTAL_EPOCHS - WARMUP_EPOCHS  # = 197
#
# all_results = []
#
# for subject_id in range(1, 21):
#     print(f"\nTraining Subject {subject_id} ...")
#

#     train_X, train_y, test_X, test_y = load_selfVR_data_cross_subject(data_path, subject_id)
#

#     train_bands = preprocess_hgd_data(train_X, fs=250, num_bands=9)  # [B, 9, C, T]
#     mean_b, std_b = band_zscore_stats(train_bands)
#
#     train_set = EEGDataset(train_X, train_y, mean=mean_b, std=std_b)
#     test_set  = EEGDataset(test_X,  test_y,  mean=mean_b, std=std_b)
#     train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
#     test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)
#
#     model = FBMSNet(nChan=32, nTime=768, nClass=2, dropoutP=0.5).to(device)
#
#     # 类权重 + CE(label smoothing)
#     weights = class_weights_from_labels(train_y, num_classes=2, device=device)
#     criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
#

#     for p in model.parameters(): p.requires_grad = False
#     for p in model.fc.parameters(): p.requires_grad = True
#     opt_head = optim.AdamW(model.fc.parameters(), lr=3e-3, weight_decay=1e-4)
#
#     for e in range(1, WARMUP_EPOCHS + 1):
#         train_loss = train_model(model, train_loader, opt_head, criterion, device)
#         acc, kappa, _, _ = evaluate_model(model, test_loader, device)
#         print(f"[HeadWarmup] Epoch {e}/{WARMUP_EPOCHS} | Loss={train_loss:.4f} | Acc={acc:.4f} | Kappa={kappa:.4f}")
#

#     for p in model.parameters(): p.requires_grad = True
#     optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAIN_EPOCHS, eta_min=3e-5)
#
#     best_acc = best_kappa = 0.0
#     best_epoch = -1
#     last_acc = last_kappa = 0.0
#
#     for epoch in range(START_MAIN, START_MAIN + MAIN_EPOCHS):  # 4..200 共197轮
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
# # Summary
# best_accs = [r['best_acc'] for r in all_results]
# best_kappas = [r['best_kappa'] for r in all_results]
# last_accs = [r['last_acc'] for r in all_results]
# last_kappas = [r['last_kappa'] for r in all_results]
#
# print("\nSummary for all subjects:")
# for r in all_results:
#     print(f"Subject {r['subject']}: Best Acc={r['best_acc']:.4f} (Epoch {r['best_epoch']}), "
#           f"Best Kappa={r['best_kappa']:.4f}, Last Acc={r['last_acc']:.4f}, Last Kappa={r['last_kappa']:.4f}")
#
# print(f"\nAverage Best Accuracy: {np.mean(best_accs):.4f}")
# print(f"Average Best Kappa: {np.mean(best_kappas):.4f}")
# print(f"Average Last Accuracy: {np.mean(last_accs):.4f}")
# print(f"Average Last Kappa: {np.mean(last_kappas):.4f}")
