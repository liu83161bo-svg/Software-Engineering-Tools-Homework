# scripts/generate_sample_data.py
import numpy as np
import pandas as pd
import os


def generate_sample_data(n_samples=20, seed=42):
    """生成模拟EEG数据样本"""
    np.random.seed(seed)

    # 年龄标签（基于您的分布）
    ages = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 26, 47]

    data = []
    for i in range(n_samples):
        # 生成随机EEG信号（1000个时间点）
        # 模拟alpha波段（8-13Hz）和beta波段（13-30Hz）活动
        t = np.linspace(0, 1, 1000)
        alpha = np.sin(2 * np.pi * 10 * t)  # 10Hz alpha
        beta = 0.5 * np.sin(2 * np.pi * 20 * t)  # 20Hz beta
        noise = 0.2 * np.random.randn(1000)  # 高斯噪声

        signal = alpha + beta + noise

        # 随机选择年龄
        age = np.random.choice(ages)

        # 生成文件名和试验索引
        subject_id = f"sub{np.random.randint(1, 50):03d}"
        session = np.random.randint(1, 4)
        file_name = f"{subject_id}_ses{session}.mat"

        data.append({
            'trial_id': i + 1000,
            'file_name': file_name,
            'trial_index': np.random.randint(0, 50),
            'age': age,
            'subject_hash': f"hash_{hash(subject_id) % 100000:05d}",
            'recording_date': f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
        })

        # 添加信号数据（前10个点作为示例）
        for j in range(min(10, len(signal))):
            data[-1][f'signal_{j}'] = signal[j]

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 确保目录存在
    os.makedirs('data', exist_ok=True)

    # 保存为CSV
    df.to_csv('data/sample_eeg_data.csv', index=False)

    # 创建完整信号的版本（JSON Lines格式）
    full_signals = []
    for i in range(n_samples):
        # 重新生成信号以确保一致性
        t = np.linspace(0, 1, 1000)
        alpha = np.sin(2 * np.pi * 10 * t)
        beta = 0.5 * np.sin(2 * np.pi * 20 * t)
        noise = 0.2 * np.random.randn(1000)
        signal = (alpha + beta + noise).tolist()

        full_signals.append({
            'trial_id': i + 1000,
            'signal': signal,
            'age': int(df.iloc[i]['age'])
        })

    # 保存为JSON Lines
    import json
    with open('data/sample_eeg_data.jsonl', 'w') as f:
        for item in full_signals:
            f.write(json.dumps(item) + '\n')

    print(f"Generated {n_samples} sample records")
    print(f"Saved to data/sample_eeg_data.csv and data/sample_eeg_data.jsonl")

    return df


if __name__ == "__main__":
    df = generate_sample_data(n_samples=20)
    print("\nSample data preview:")
    print(df.head())