import os
import numpy as np
import scipy.io as scio
import mne
from mne.io import read_raw_gdf
import argparse

def convert_gdf_to_mat(gdf_file_path, label_file_path, output_dir):
    """
    将单个GDF文件转换为MAT文件，使用真实标签文件
    
    Args:
        gdf_file_path: GDF文件路径
        label_file_path: 对应的标签文件路径
        output_dir: 输出目录
    """
    # 读取GDF文件
    raw = read_raw_gdf(gdf_file_path, preload=True, verbose=False)
    
    # 读取真实标签
    label_data = scio.loadmat(label_file_path)
    true_labels = label_data['classlabel'].flatten()  # shape: (288,)
    
    # 选择22个EEG通道（排除EOG通道）
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG-') and not ch.startswith('EOG')]
    if len(eeg_channels) != 22:
        print(f"警告: 找到{len(eeg_channels)}个EEG通道，期望22个")
        print(f"EEG通道: {eeg_channels}")
    
    raw.pick_channels(eeg_channels)
    print(f"选择的EEG通道: {len(raw.ch_names)}个")
    print("注意：滤波将在dataload.py中统一处理")
    
    # 获取数据和事件
    data = raw.get_data()  # shape: (channels, time_points)
    events, event_id = mne.events_from_annotations(raw)
    
    # 提取试验数据
    # BCI Competition IV 2a的事件编码：
    # 769: 左手运动想象开始
    # 770: 右手运动想象开始  
    # 771: 脚部运动想象开始
    # 772: 舌头运动想象开始
    # 783: 提示音
    
    # 找到运动想象开始的事件
    # 注意：MNE会重新映射事件编码，769->7, 770->8, 771->9, 772->10
    mi_events = events[np.isin(events[:, 2], [7, 8, 9, 10])]
    
    # 参数设置
    sfreq = raw.info['sfreq']  # 采样率
    trial_length = int(4 * sfreq)  # 4秒试验长度
    
    # 提取试验数据
    trials = []
    labels = []
    
    print(f"开始提取 {len(mi_events)} 个试验...")
    
    # 检查事件数量与标签数量是否匹配
    if len(mi_events) != len(true_labels):
        print(f"⚠️ 警告: 事件数量({len(mi_events)})与标签数量({len(true_labels)})不匹配!")
        print(f"将使用前{min(len(mi_events), len(true_labels))}个标签")
    
    for i, event in enumerate(mi_events):
        start_sample = event[0]
        end_sample = start_sample + trial_length
        
        if end_sample <= data.shape[1] and i < len(true_labels):
            trial_data = data[:, start_sample:end_sample]
            trials.append(trial_data)
            
            # 使用真实标签
            labels.append(true_labels[i])
        else:
            print(f"跳过试验 {i+1}: 开始={start_sample}, 结束={end_sample}, 数据长度={data.shape[1]}, 标签索引={i}")
    
    print(f"成功提取 {len(trials)} 个试验")
    
    # 显示最终的标签分布
    if len(labels) > 0:
        final_labels = np.array(labels)
        print(f"最终标签分布: {np.bincount(final_labels)}")
    
    if len(trials) == 0:
        raise ValueError("没有成功提取到任何试验数据")
    
    # 转换为numpy数组
    try:
        EEG_data = np.array(trials)  # shape: (trials, channels, time_points)
        label = np.array(labels)     # shape: (trials,)
        print(f"数组转换成功: EEG_data {EEG_data.shape}, labels {label.shape}")
    except Exception as e:
        print(f"数组转换失败: {e}")
        print(f"试验列表长度: {len(trials)}")
        if trials:
            print(f"第一个试验形状: {trials[0].shape}")
            print(f"最后一个试验形状: {trials[-1].shape}")
        raise
    
    # 转换维度为 (channels, time_points, trials)
    EEG_data = EEG_data.transpose(1, 2, 0)
    
    # 保存为MAT文件
    filename = os.path.basename(gdf_file_path).replace('.gdf', '.mat')
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    
    scio.savemat(output_path, {
        'EEG_data': EEG_data,
        'label': label
    })
    
    print(f"转换完成: {gdf_file_path} -> {output_path}")
    print(f"使用标签文件: {label_file_path}")
    print(f"数据形状: EEG_data {EEG_data.shape}, label {label.shape}")
    print(f"标签分布: {np.bincount(label)}")
    
    return output_path

def convert_bci2a_dataset(gdf_root_dir, label_root_dir, mat_root_dir):
    """
    转换整个BCI Competition IV 2a数据集
    
    Args:
        gdf_root_dir: GDF文件根目录
        label_root_dir: 标签文件根目录
        mat_root_dir: MAT文件输出根目录
    """
    if not os.path.exists(mat_root_dir):
        os.makedirs(mat_root_dir)
    
    # 遍历所有受试者
    for subject_id in range(1, 10):  # A01-A09
        subject_folder = f"A{subject_id:02d}"
        subject_mat_dir = os.path.join(mat_root_dir, subject_folder)
        
        if not os.path.exists(subject_mat_dir):
            os.makedirs(subject_mat_dir)
        
        # 转换训练和测试文件
        for session in ['T', 'E']:  # T=训练, E=评估
            gdf_file = os.path.join(gdf_root_dir, f"A{subject_id:02d}{session}.gdf")
            label_file = os.path.join(label_root_dir, f"A{subject_id:02d}{session}.mat")
            
            if os.path.exists(gdf_file) and os.path.exists(label_file):
                try:
                    # 根据文件类型确定输出文件名
                    if session == 'T':
                        output_file = os.path.join(subject_mat_dir, "training.mat")
                    else:
                        output_file = os.path.join(subject_mat_dir, "evaluation.mat")
                    
                    # 转换文件
                    temp_file = convert_gdf_to_mat(gdf_file, label_file, subject_mat_dir)
                    
                    # 重命名文件以匹配dataload.py的期望格式
                    if os.path.exists(temp_file):
                        os.rename(temp_file, output_file)
                        print(f"重命名: {temp_file} -> {output_file}")
                        
                except Exception as e:
                    print(f"转换失败 {gdf_file}: {str(e)}")
            else:
                if not os.path.exists(gdf_file):
                    print(f"GDF文件不存在: {gdf_file}")
                if not os.path.exists(label_file):
                    print(f"标签文件不存在: {label_file}")

def main():
    parser = argparse.ArgumentParser(description='将BCI Competition IV 2a的GDF文件转换为MAT文件')
    parser.add_argument('--gdf_dir', type=str, 
                       default='../data/BCICIV_2a_gdf/BCICIV_2a_gdf',
                       help='GDF文件根目录')
    parser.add_argument('--label_dir', type=str,
                       default='../data/BCICIV_2a_gdf/2a_true_labels',
                       help='标签文件根目录')
    parser.add_argument('--mat_dir', type=str,
                       default='../data/BCICIV_2a_mat',
                       help='MAT文件输出根目录')
    parser.add_argument('--single_gdf', type=str, default=None,
                       help='转换单个GDF文件')
    parser.add_argument('--single_label', type=str, default=None,
                       help='对应的标签文件')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='单个文件转换时的输出目录')
    
    args = parser.parse_args()
    
    if args.single_gdf and args.single_label:
        # 转换单个文件
        convert_gdf_to_mat(args.single_gdf, args.single_label, args.output_dir)
    elif args.single_gdf:
        print("错误：转换单个文件时必须同时提供GDF文件和标签文件")
    else:
        # 转换整个数据集
        convert_bci2a_dataset(args.gdf_dir, args.label_dir, args.mat_dir)
        print("数据集转换完成！")

if __name__ == "__main__":
    main() 