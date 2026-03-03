# 哑铃弯举教练模型 POC

## 数据集

| 文件 | 用途 |
|------|------|
| MyoGym.csv | Phase 1 EMG：八通道，50Hz，小臂采集 |
| imu_fatigue.csv | Phase 1 IMU：六轴，50Hz，手腕采集 |
| j-tom03.csv | Phase 2 原始：单通道 EMG (2148Hz) + 六轴 IMU (370Hz)，肱二头肌/手腕 |
| coaching.jsonl | Phase 3 标注：每行含 tags（语义标签）和 text（人工点评），作为训练 GT 与输出风格参考 |
| analyse_jtom.csv | Phase 3 Layer2：由 scripts/analyse_jtom.py 从 j-tom03 生成，含借力指数等硬核指标 |

预处理：`j-tom03.csv` → `scripts/preprocess_jtom.py` → `dataset/processed/jtom_integrated_samples.pt`

---

## Phase 1：信号表征学习

- **任务**：让模型理解什么是「电信号」和「加速度信号」
- **输入**：MyoGym (MyoGym.csv) + Fatigue IMU (imu_fatigue.csv)
- **核心**：SSL 自监督学习，训练 Encoder（CNN+Transformer）  
  → `models/encoder.py`、`models/ssl_model.py`、`training/ssl_trainer.py`
- **产出**：预训练 Encoder，可从原始波形提取高维特征向量

---

## Phase 2：物理状态融合

- **任务**：多模态对齐，研究 EMG（内部发力）与 IMU（外部动作）的关联，对齐 RPE/重量
- **输入**：J-tom03 预处理后的 EMG+IMU（`dataset/processed/jtom_integrated_samples.pt`）
- **核心**：FusionLayer（交互注意力），学习何时看 EMG（爆发力）、何时看 IMU（身体晃动）  
  → `models/fusion.py`、`training/fusion_trainer.py`
- **产出**：强力的「运动特征提取器」，输出向量蕴含动作标准度、疲劳度等物理含义

---

## Phase 3：语义重编程

- **任务**：跨模态转换（物理空间 → 语言空间）
- **核心组件**：
  - **scripts/extract.py**：从 coaching.jsonl 提取「语义种子」(Prototypes)，初始化 ReprogrammingLayer
  - **ReprogrammingLayer**：把 Phase 2 物理向量映射成 Qwen-7B 能理解的 Soft Tokens  
    → `models/reprogramming.py`、`models/fusionllm.py`、`training/phase3_trainer.py`
  - **三层 Prompt**：Layer1 身份、Layer2 硬核指标 (analyse_jtom.csv)、Layer3 Soft Tokens
- **产出**：`outputs/phase3_preds.jsonl`，生成与 coaching.jsonl 风格一致的健身点评

---

## 运行方式

```bash
# Phase 1
python main.py phase1 --modality both   # 或 emg / imu

# Phase 2（需先有 Phase 1 的 encoder checkpoint）
python main.py phase2 --auto_preprocess   # 若 jtom_integrated_samples.pt 不存在则自动预处理

# Phase 3（需先有 Phase 1 + Phase 2 checkpoint + prototypes_init.pt）
python main.py phase3
```

**可选脚本**：
- `scripts/extract.py`：生成 `dataset/prototypes_init.pt`，供 Phase 3 重编程层初始化
- `scripts/analyse_jtom.py`：从 j-tom03.csv 生成 `dataset/analyse_jtom.csv`
- `scripts/infer_phase3.py`：推理生成 outputs/phase3_preds.jsonl
- `scripts/eval_phase3.py`：评估 Phase 3 文本输出（标签匹配、badcase 等）
- `scripts/run_phase2_loo.py`：Phase 2 留一人交叉验证

配置见 `config.yaml`。

---

## 项目结构

```
dataloaders/    # 数据加载（dataloaders 为代码包，dataset/ 为数据文件）
├── ssl_dataset.py      # Phase1 Dataset + create_ssl_dataloaders
├── phase2_dataset.py   # Phase2 Dataset + create_phase2_dataloaders
├── phase3_dataset.py   # Phase3 Dataset
├── create_dataset.py   # 兼容层，聚合导出
└── augmentations.py    # 数据增强
models/         # 模型定义
training/       # 训练器
scripts/        # 预处理、推理、评估脚本
```




