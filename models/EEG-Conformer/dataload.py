import os
import argparse
import torch
from collections import OrderedDict
import logging
import pickle
import scipy.io as scio
from scipy import signal
import numpy as np
import mne


log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(level=logging.INFO)

def load_bciciv2a_data_single_subject(data_path, subject_id):
    subject = f"A{subject_id:02d}"
    # 加载训练数据和标签
    train_X = np.load(os.path.join(data_path, f"{subject}T_data.npy"))
    train_Y = np.load(os.path.join(data_path, f"{subject}T_label.npy"))-1

    # 加载测试数据和标签
    test_X = np.load(os.path.join(data_path, f"{subject}E_data.npy"))
    test_Y = np.load(os.path.join(data_path, f"{subject}E_label.npy"))-1

    test_Y = torch.tensor(test_Y, dtype=torch.int64).squeeze(-1)  # 将 (288, 1) 转换为 (288,)

    # 应用巴特沃斯带通滤波器 (0.5 Hz - 40 Hz, fs=250 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)
    filtered_train_signal = signal.lfilter(b, a, train_X, axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X, axis=-1)

    # 转换为 PyTorch 张量
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y

def load_selfImage_single_subject(data_path, subject_id):
    """
    加载BCI Competition IV Dataset 自采视频的npy格式数据
    数据格式: I{subject_id:02d}T_data.npy 和 I{subject_id:02d}T_label.npy

    Args:
        data_path: 数据路径， 'selfdata/image/'
        subject_id: 受试者ID (1-20)

    Returns:
        filtered_train_signal, train_data_Y, filtered_test_signal, test_data_Y
    """
    # 构建文件路径 - 自采视频数据集采用I开头的命名
    subject = f"I{subject_id:02d}"

    # 加载训练数据和标签 (只有T文件，没有E文件)
    train_data_path = os.path.join(data_path, f"{subject}T_data.npy")
    train_label_path = os.path.join(data_path, f"{subject}T_label.npy")
    test_data_path = os.path.join(data_path, f"{subject}E_data.npy")
    test_label_path = os.path.join(data_path, f"{subject}E_label.npy")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"找不到训练数据文件: {train_data_path}")
    if not os.path.exists(train_label_path):
        raise FileNotFoundError(f"找不到训练标签文件: {train_label_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"找不到训练数据文件: {test_data_path}")
    if not os.path.exists(test_label_path):
        raise FileNotFoundError(f"找不到训练标签文件: {test_label_path}")
    # 加载数据
    train_data_X = np.load(train_data_path)  # 形状: (n_trials, n_channels, n_samples)
    train_data_Y = np.load(train_label_path)  # 形状: (n_trials,)
    test_data_X = np.load(test_data_path)  # 形状: (n_trials, n_channels, n_samples)
    test_data_Y = np.load(test_label_path)  # 形状: (n_trials,)
    print(f"加载受试者 {subject}: 训练数据形状 {train_data_X.shape}, 训练标签形状 {train_data_Y.shape}")
    print(f"加载受试者 {subject}: 测试数据形状 {train_data_X.shape}, 测试标签形状 {train_data_Y.shape}")
    print(f"标签唯一值: {np.unique(train_data_Y)}")

    # # 转换为 PyTorch 张量
    train_X = torch.tensor(train_data_X, dtype=torch.float32)
    train_Y = torch.tensor(train_data_Y, dtype=torch.int64).view(-1)
    test_X = torch.tensor(test_data_X, dtype=torch.float32)
    test_Y = torch.tensor(test_data_Y, dtype=torch.int64).view(-1)

    # 应用巴特沃斯带通滤波器 (0.5 Hz - 40 Hz, fs=250 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)

    filtered_train_signal = signal.lfilter(b, a, train_X,axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X,axis=-1)

    # 转换回 PyTorch 张量
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    print(f"训练集: {filtered_train_signal.shape}, 测试集: {filtered_test_signal.shape}")

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y
def load_selfVedio_single_subject(data_path, subject_id):

    # 构建文件路径 - 自采视频数据集采用V开头的命名
    subject = f"V{subject_id:02d}"

    # 加载训练数据和标签 (只有T文件，没有E文件)
    train_data_path = os.path.join(data_path, f"{subject}T_data.npy")
    train_label_path = os.path.join(data_path, f"{subject}T_label.npy")
    test_data_path = os.path.join(data_path, f"{subject}E_data.npy")
    test_label_path = os.path.join(data_path, f"{subject}E_label.npy")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"找不到训练数据文件: {train_data_path}")
    if not os.path.exists(train_label_path):
        raise FileNotFoundError(f"找不到训练标签文件: {train_label_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"找不到训练数据文件: {test_data_path}")
    if not os.path.exists(test_label_path):
        raise FileNotFoundError(f"找不到训练标签文件: {test_label_path}")
    # 加载数据
    train_data_X = np.load(train_data_path)  # 形状: (n_trials, n_channels, n_samples)
    train_data_Y = np.load(train_label_path)  # 形状: (n_trials,)
    test_data_X = np.load(test_data_path)  # 形状: (n_trials, n_channels, n_samples)
    test_data_Y = np.load(test_label_path)  # 形状: (n_trials,)
    print(f"加载受试者 {subject}: 训练数据形状 {train_data_X.shape}, 训练标签形状 {train_data_Y.shape}")
    print(f"加载受试者 {subject}: 测试数据形状 {train_data_X.shape}, 测试标签形状 {train_data_Y.shape}")
    print(f"标签唯一值: {np.unique(train_data_Y)}")

    # # 转换为 PyTorch 张量
    train_X = torch.tensor(train_data_X, dtype=torch.float32)
    train_Y = torch.tensor(train_data_Y, dtype=torch.int64).view(-1)
    test_X = torch.tensor(test_data_X, dtype=torch.float32)
    test_Y = torch.tensor(test_data_Y, dtype=torch.int64).view(-1)

    # 应用巴特沃斯带通滤波器 (0.5 Hz - 40 Hz, fs=250 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)

    filtered_train_signal = signal.lfilter(b, a, train_X,axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X,axis=-1)

    # 转换回 PyTorch 张量
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    print(f"训练集: {filtered_train_signal.shape}, 测试集: {filtered_test_signal.shape}")

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y
def load_selfVR_single_subject(data_path, subject_id):

    # 构建文件路径 - 自采视频数据集采用VR开头的命名
    subject = f"VR{subject_id:02d}"

    # 加载训练数据和标签 (只有T文件，没有E文件)
    train_data_path = os.path.join(data_path, f"{subject}T_data.npy")
    train_label_path = os.path.join(data_path, f"{subject}T_label.npy")
    test_data_path = os.path.join(data_path, f"{subject}E_data.npy")
    test_label_path = os.path.join(data_path, f"{subject}E_label.npy")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"找不到训练数据文件: {train_data_path}")
    if not os.path.exists(train_label_path):
        raise FileNotFoundError(f"找不到训练标签文件: {train_label_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"找不到训练数据文件: {test_data_path}")
    if not os.path.exists(test_label_path):
        raise FileNotFoundError(f"找不到训练标签文件: {test_label_path}")
    # 加载数据
    train_data_X = np.load(train_data_path)  # 形状: (n_trials, n_channels, n_samples)
    train_data_Y = np.load(train_label_path)  # 形状: (n_trials,)
    test_data_X = np.load(test_data_path)  # 形状: (n_trials, n_channels, n_samples)
    test_data_Y = np.load(test_label_path)  # 形状: (n_trials,)
    print(f"加载受试者 {subject}: 训练数据形状 {train_data_X.shape}, 训练标签形状 {train_data_Y.shape}")
    print(f"加载受试者 {subject}: 测试数据形状 {train_data_X.shape}, 测试标签形状 {train_data_Y.shape}")
    print(f"标签唯一值: {np.unique(train_data_Y)}")

    # # 转换为 PyTorch 张量
    train_X = torch.tensor(train_data_X, dtype=torch.float32)
    train_Y = torch.tensor(train_data_Y, dtype=torch.int64).view(-1)
    test_X = torch.tensor(test_data_X, dtype=torch.float32)
    test_Y = torch.tensor(test_data_Y, dtype=torch.int64).view(-1)

    # 应用巴特沃斯带通滤波器 (0.5 Hz - 40 Hz, fs=250 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)

    filtered_train_signal = signal.lfilter(b, a, train_X,axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X,axis=-1)

    # 转换回 PyTorch 张量
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    # print(f"训练集: {filtered_train_signal.shape}, 测试集: {filtered_test_signal.shape}")

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y

def load_HGD_single_subject(data_path, subject_id):
    import os
    from scipy import signal

    subject = f"H{subject_id:02d}"
    train_X = np.load(os.path.join(data_path, f"{subject}T_data.npy"))
    train_Y = np.load(os.path.join(data_path, f"{subject}T_label.npy")).squeeze()  # 加squeeze

    test_X = np.load(os.path.join(data_path, f"{subject}E_data.npy"))
    test_Y = np.load(os.path.join(data_path, f"{subject}E_label.npy")).squeeze()    # 加squeeze

    # 转成int64并保证为1D标签
    train_Y = train_Y.astype(np.int64)
    test_Y = test_Y.astype(np.int64)

    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)
    filtered_train_signal = signal.lfilter(b, a, train_X, axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X, axis=-1)

    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y


#=========================leaveone===========================

def load_bciciv2a_data_cross_subject(data_path, subject_id):
    """
    留一交叉被试 (LOSO) 数据加载（BCI IV 2a）
    参数保持与原单受试者加载一致： (data_path, subject_id)
    返回：
        train_X: torch.float32, 形如 (N_train, C, T) 或 (N_train, ..., T)
        train_y: torch.int64,   形如 (N_train,)
        test_X : torch.float32, 形如 (N_test,  C, T) 或 (N_test,  ..., T)
        test_y : torch.int64,   形如 (N_test,)
    规则：
        - 训练集：除 subject_id 外每个受试者的 (train+test) 先合并，再做上采样增强；
        - 测试集：subject_id 的 (train+test) 合并，不做增强；
        - 增强后/合并后统一做 0.5–40 Hz 带通滤波（fs=250）。
    """
    # ===== 可调参数 =====
    aug_ratio = 0.10   # 上采样增强比例；如 0.10 表示对每个训练受试者合并后的样本再增加 10%
    fs = 250           # 采样率，用于设计巴特沃斯滤波器
    band = (0.5, 40)   # 带通范围
    subject_ids = list(range(1, 10))  # A01~A09

    # ===== 内部小工具：加载原始 numpy，不做增强/滤波 =====
    def _load_raw_numpy(sid):
        subj = f"A{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy")) - 1
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy")) - 1
        # 兼容 (N,1) 标签
        if trY.ndim > 1: trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1: teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # ===== 设计滤波器（统一使用）=====
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    # ===== 训练集：其余受试者（先合并，再增强，再滤波）=====
    train_ids = [sid for sid in subject_ids if sid != subject_id]
    train_X_list, train_y_list = [], []
    for sid in train_ids:
        trX, trY, teX, teY = _load_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        X_all = signal.lfilter(b, a, X_all, axis=-1)

        # —— 转 Tensor 累加 ——
        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # ===== 测试集：留出受试者（合并但不增强；同样滤波）=====
    trX, trY, teX, teY = _load_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)
    X_te = signal.lfilter(b, a, X_te, axis=-1)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y

def load_HGD_data_cross_subject(data_path, subject_id):
    """
    HGD 留一被试 (LOSO) 数据加载（无额外参数）
    - 受试者固定范围：1..14（range(1, 15)）
    - 训练集：其余受试者 (train+test) 合并 → 上采样增强(10%) → 0.5–40 Hz 带通滤波(fs=250)
    - 测试集：subject_id 的 (train+test) 合并 → 不增强 → 同样滤波
    返回:
        train_X: torch.float32, (N_train, C, T)
        train_y: torch.int64,   (N_train,)
        test_X : torch.float32, (N_test,  C, T)
        test_y : torch.int64,   (N_test,)
    """
    # 固定参数（不对外暴露）
    aug_ratio = 0.10      # 上采样增强比例
    fs = 250              # 采样率（用于滤波）
    band = (0.5, 40)      # 带通范围
    time_axis = -1        # 假设时间维在最后 (N, C, T)

    # 受试者列表与合法性检查
    subject_ids = list(range(1, 15))  # 1..14
    if subject_id not in subject_ids:
        raise ValueError(f"subject_id={subject_id} 不在有效范围 1..14")
    train_ids = [sid for sid in subject_ids if sid != subject_id]

    # 统一设计滤波器
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    # 工具：加载单被试原始 numpy（不做增强/滤波）
    def _load_hgd_raw_numpy(sid):
        subj = f"H{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy"))
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy"))
        if trY.ndim > 1: trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1: teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # ===== 训练集：合并 → 增强 → 滤波 =====
    train_X_list, train_y_list = [], []
    for sid in train_ids:
        trX, trY, teX, teY = _load_hgd_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        # 统一带通滤波
        X_all = signal.lfilter(b, a, X_all, axis=time_axis)

        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # ===== 测试集：合并 → 不增强 → 滤波 =====
    trX, trY, teX, teY = _load_hgd_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)
    X_te = signal.lfilter(b, a, X_te, axis=time_axis)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y

def load_selfVR_data_cross_subject(data_path, subject_id):
    """
    VR-MI 留一被试 (LOSO) 数据加载（无额外参数）
    - 被试范围固定为 1..20（range(1, 21)）
    - 训练集：其余被试的 (train+test) 合并 → 上采样增强(10%) → 0.5–40 Hz 带通滤波(fs=250)
    - 测试集：subject_id 的 (train+test) 合并 → 不增强 → 同样滤波
    返回:
        train_X: torch.float32, (N_train, C, T)
        train_y: torch.int64,   (N_train,)
        test_X : torch.float32, (N_test,  C, T)
        test_y : torch.int64,   (N_test,)
    约定文件命名：VRxxT_data.npy / VRxxT_label.npy / VRxxE_data.npy / VRxxE_label.npy
    """
    # 固定参数（不对外暴露）
    aug_ratio = 0.10
    fs = 250
    band = (0.5, 40)
    time_axis = -1  # 假设时间维在最后 (N, C, T)

    # 被试列表与合法性检查
    subject_ids = list(range(1, 21))  # 1..20
    if subject_id not in subject_ids:
        raise ValueError(f"subject_id={subject_id} 不在有效范围 1..20")
    train_ids = [sid for sid in subject_ids if sid != subject_id]

    # 统一设计滤波器
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    # 工具：加载单被试原始 numpy（不做增强/滤波）
    def _load_vr_raw_numpy(sid):
        subj = f"VR{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy"))
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy"))
        if trY.ndim > 1: trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1: teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # ===== 训练集：合并 → 增强 → 滤波 =====
    train_X_list, train_y_list = [], []
    for sid in train_ids:
        trX, trY, teX, teY = _load_vr_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        # 统一带通滤波
        X_all = signal.lfilter(b, a, X_all, axis=time_axis)

        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # ===== 测试集：合并 → 不增强 → 滤波 =====
    trX, trY, teX, teY = _load_vr_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)
    X_te = signal.lfilter(b, a, X_te, axis=time_axis)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y