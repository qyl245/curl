import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# --- 1. 参数配置 ---
# 注意：基于你之前的合并逻辑，CSV中的行数应是以EMG频率(2148.1481Hz)为准对齐的
FS_EMG = 2148.1481
TARGET_FS = 2148.1481  # 基准采样率


def butter_lowpass_filter(data, cutoff, fs, btype='low', order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, data)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


# --- 2. 预处理函数 ---
def preprocess_dataset(df):
    print("正在进行信号预处理（滤波与包络提取）...")
    # EMG 处理：带通滤波 -> 取绝对值(整流) -> 低通滤波(包络)
    emg_bp = butter_bandpass_filter(df['emg'], 20, 450, FS_EMG)
    df['emg_env'] = butter_lowpass_filter(np.abs(emg_bp), 5, FS_EMG)

    # IMU 处理：低通滤波去除高频抖动噪声（保留 10Hz 以下的人体动作）
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in imu_cols:
        df[col] = butter_lowpass_filter(df[col], 10, FS_EMG)
    return df


# --- 3. 单个动作特征提取函数 ---
def extract_rep_features(group):
    # 时间维度
    duration = len(group) / FS_EMG

    acc_y = group['acc_y'].values
    jerk = np.diff(acc_y).std() * FS_CSV  # 衡量加速度的变化剧烈程度

    std_x = group['acc_x'].std()
    std_y = group['acc_y'].std()
    std_z = group['acc_z'].std()
    # 借力指数 = (横向+纵向) / 主轴能量
    cheating_index = (std_x + std_z) / (std_y + 1e-6)

    # 3. Tremor (高频震颤)
    # 对陀螺仪信号做高通滤波(>10Hz)，保留震颤部分
    gyro_raw = group['gyro_y'].values
    gyro_high = gyro_raw - butter_lowpass_filter(gyro_raw, 10, FS_CSV, 'low')
    tremor_score = np.std(gyro_high)


    # 强度维度：EMG包络均值
    emg_env = group['emg_env'].values
    peak_idx = np.argmax(emg_env) / len(emg_env) # 峰值位置(0-1之间)

    return pd.Series({
        'duration': duration,
        'jerk_score': jerk,
        'cheating_index': cheating_index,
        'tremor_score': tremor_score,
        'emg_intensity': np.mean(emg_env),
        'peak_location': peak_idx,
        'rpe': group['rpe'].iloc[0]
    })


# --- 4. 语义映射逻辑 ---
def generate_coach_tags(df_reps):
    final_results = []
    # 按每个人、每一组进行独立对比
    for (pid, weight, cls), group in df_reps.groupby(['person_id', 'weight', 'class']):

        # 【关键修正】：添加 numeric_only=True，跳过字符串 ID
        if len(group) >= 2:
            baseline = group.iloc[:2].mean(numeric_only=True)
        else:
            baseline = group.iloc[0].to_dict()  # 如果只有一次动作，就以自己为基准

        for i, row in group.iterrows():
            tags = []

            # A. 节奏与平滑度 (Jerk)
            if row['jerk_score'] > baseline['jerk_score'] * 1.4:
                tags.append("动作流畅度下降")
            if row['tremor_score'] > baseline['tremor_score'] * 1.8:
                tags.append("明显肌肉震颤")

            # B. 代偿分析 (Cheating)
            if row['cheating_index'] > baseline['cheating_index'] * 1.3:
                tags.append("出现躯干借力/晃动")

            # C. 动作完整性
            if row['duration'] > baseline['duration'] * 1.3:
                tags.append("向心收缩挣扎")

            # D. 发力点偏移
            if row['peak_location'] < 0.3:  # 发力峰值过早
                tags.append("爆发式借力")
            elif row['peak_location'] > 0.7:
                tags.append("后程发力困难")

            # E. 综合评价
            if not tags:
                tags.append("动作教科书级")

            res = row.to_dict()
            res['semantic_tags'] = " / ".join(tags)
            res['person_id'], res['weight'], res['class'] = pid, weight, cls
            final_results.append(res)

    return pd.DataFrame(final_results)


# --- 5. 执行主程序 ---
if __name__ == "__main__":
    # 加载已对齐的 CSV
    df = pd.read_csv('dataset/j-tom03.csv')

    # 预处理
    df_clean = preprocess_dataset(df)

    # 提取特征 (按动作切片)
    print("按 Rep 提取特征指标...")
    rep_metrics = df_clean.groupby(['person_id', 'weight', 'class', 'rep_num']).apply(
        extract_rep_features,
        include_groups=False
    ).reset_index()

    # 映射语义
    print("映射语义标签...")
    final_coaching_data = generate_coach_tags(rep_metrics)

    # 保存结果（与 config.yaml / phase3 期望路径一致）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, 'dataset', 'analyse_jtom.csv')
    final_coaching_data.to_csv(out_path, index=False)
    print(f"分析完成！已保存至 {out_path}")
