哑铃弯举教练模型 POC
Phase 1：信号表征学习
● 任务：让模型理解什么是“电信号”和“加速度信号”
● 输入：MyoGym的二头肌相关动作（对应dataset里的MyoGym.csv，八通道emg，采样频率为50Hz，从小臂采集）,Fatigue的IMU数据（对应dataset里的imu_fatigue.csv，六轴imu，从手腕采集，采样频率为50Hz）
● 核心：通过 SSL（自监督学习）训练 Encoder（CNN+Transformer） （涉及encoder.py、ssl_model.py、ssl_trainer.py等）
● 产出：能够从原始波形中提取高维特征向量的预训练 Encoder

Phase 2：物理状态融合 
● 任务：多模态对齐。研究 EMG（内部发力）和 IMU（外部动作）是如何关联的，并对齐到物理指标（RPE/重量）。
● 输入：J-tom03的emg和imu（emg从肱二头肌采集，单通道，采样频率为2148.1481 Hz；imu从手腕采集，六轴，采样频率为370.3704Hz）
● 核心：FusionLayer（交互注意力机制）。它学习在什么时候看 EMG（看爆发力），什么时候看 IMU（看身体晃动） （涉及fusion.py、fusion_trainer.py等）
● 产出：一个强力的“运动特征提取器”，它输出的向量已经包含了这个动作是否标准、是否疲劳的物理含义。

Phase 3：语义重编程
● 任务：跨模态转换（从物理空间跨越到语言空间）
● 核心组件：（涉及reprogramming.py、fusionllm.py、phase3_trainer.py等）
  a. extract.py：从文本中提取“语义种子”（Prototypes），告诉模型“借力”在 LLM 的脑子里长什么样。
  b. ReprogrammingLayer：这是唯一的训练点。它像一个翻译官，把 Phase 2 的物理向量映射成 Qwen-7B 能听懂的“Soft Tokens”。
  c. 三层 Prompt 结构：
    ■ Layer 1 (System): 赋予模型身份（你是教练）。
    ■ Layer 2 (Stats): 喂入 analyse_jtom.csv 的硬核指标（借力指数等），防止 LLM 瞎编。
    ■ Layer 3 (Soft Tokens): 喂入重编程后的信号特征，提供现场感。
● 产出：coaching.jsonl 中那样有温度、有专业度的健身点评。