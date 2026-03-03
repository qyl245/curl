import pandas as pd
import numpy as np
import torch
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """带通滤波 - 滤除低频运动伪影和高频噪声"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # 确保频率不超出奈奎斯特频率限制
    if high >= 1.0: high = 0.99
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def extract_envelope(data, fs, lowcut_envelope=5.0):
    """包络提取 - 提取肌肉活动强度趋势"""
    # 1. 整流 (取绝对值)
    rectified = np.abs(data)
    # 2. 低通滤波 (通常在 5-10Hz 之间提取包络)
    nyq = 0.5 * fs
    low = lowcut_envelope / nyq
    b, a = signal.butter(4, low, btype='low')
    envelope = signal.filtfilt(b, a, rectified)
    return envelope

def lowpass_filter(data, cutoff, fs, order=4):
    """低通滤波 - 用于 IMU 降噪"""
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = signal.butter(order, low, btype='low')
    return signal.filtfilt(b, a, data)

def preprocess_jtom_integrated(csv_path, save_path, target_length=200):
    """
    整合后的预处理流程：归一化 -> 滤波 -> 包络提取 -> 重采样
    """
    print(f"开始处理数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 采样频率定义 (根据原始设备参数)
    fs_emg = 2148.1481
    fs_imu = 370.3704
    
    # 列名定义
    emg_cols = ['emg']
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    processed_samples = []
    
    # 1. 全局处理：按人进行 Z-score 归一化 (消除个体差异)
    print("正在进行按人维度的 Z-score 归一化...")
    for person in df['person_id'].unique():
        person_mask = df['person_id'] == person
        scaler = StandardScaler()
        # 针对该人员的所有信号列进行标准化
        df.loc[person_mask, emg_cols + imu_cols] = scaler.fit_transform(df.loc[person_mask, emg_cols + imu_cols])
        
    # 2. 细粒度处理：按 person + weight + class + rep_num 切割与信号增强
    # 与 coaching/analyse_jtom 的 signal_id 口径保持一致，避免 phase3 对齐缺失。
    print("正在进行信号增强与重采样...")
    # 使用 groupby 提高效率
    grouped = df.groupby(['person_id', 'weight', 'class', 'rep_num'])
    
    for (person_id, weight, class_id, rep_id), group_df in grouped:
        # 剔除异常短的片段 (至少需要能做滤波的长度)
        if len(group_df) < 50:
            continue
            
        # --- A. EMG 信号增强 ---
        raw_emg = group_df['emg'].values
        # 带通滤波 (20-450Hz 黄金频段)
        filtered_emg = butter_bandpass_filter(raw_emg, 20.0, 450.0, fs_emg)
        # 提取包络 (反映肌肉收缩强度)
        envelope_emg = extract_envelope(filtered_emg, fs_emg)
        
        # --- B. IMU 信号降噪 ---
        imu_signals = []
        for col in imu_cols:
            # 20Hz 低通滤波，保留人体动作主频，滤除高频震动
            filtered_imu = lowpass_filter(group_df[col].values, 20.0, fs_imu)
            imu_signals.append(filtered_imu)
        imu_signals = np.array(imu_signals) # [6, len]
        
        # --- C. 长度统一化 (Resampling) ---
        # 将信号统一缩放到 target_length (例如 200)
        emg_final = signal.resample(envelope_emg, target_length)
        imu_final = np.array([signal.resample(sig, target_length) for sig in imu_signals])
        
        # --- D. 封装为 PyTorch 样本 ---
        sample = {
            'emg': torch.FloatTensor(emg_final).unsqueeze(0), # 形状: [1, 200]
            'imu': torch.FloatTensor(imu_final),              # 形状: [6, 200]
            'label': {
                'rpe': int(group_df['rpe'].iloc[0]),
                'person_id': person_id,
                'weight': float(weight),
                'class': class_id,
                'rep_num': int(rep_id)
            }
        }
        processed_samples.append(sample)

    # 3. 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(processed_samples, save_path)
    print(f"✅ 整合预处理完成！")
    print(f"总计生成样本数: {len(processed_samples)}")
    print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    # 配置路径
    INPUT_CSV = 'dataset/j-tom03.csv'
    OUTPUT_PT = 'dataset/processed/jtom_integrated_samples.pt'
    
    preprocess_jtom_integrated(INPUT_CSV, OUTPUT_PT)