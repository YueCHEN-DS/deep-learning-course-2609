# Hotel Review Aspect-Based Sentiment Analysis (MLP & Web Studio)

本项目为酒店评论细粒度（8个方面 × 5种情感状态）多任务情感分析系统，包含完整的 **数据标注与清洗集、PyTorch MLP 训练流水线、四代模型演进基准、FastAPI 后端服务** 以及 **现代深色玻璃拟态 Web 交互界面**（含多句评论分析、单句探针探测、四代模型演进看板、训练数据可视化与在线修正重训工作台）。

---

## 目录文件结构

```
├── Hotel_Review_MLP_Project.pptx / .pdf   # 项目说明课件 (36页)
├── README.md                              # 项目说明文档
├── requirements.txt                       # Python 依赖清单
├── .gitignore                             # Git 忽略配置（防泄露密钥与缓存）
├── api_keys.example.txt                   # API Key 配置模版（复制为 api_keys.txt 使用）
│
├── hotel_sentences_2000.csv               # 2,000 条训练候选句子原文、上下文与元数据
├── hotel_embeddings_256.npz               # 2,000 条句子的 256 维 Qwen 向量及 ID 索引
├── hotel_demo_reviews_30.csv              # 30 条独立的长篇酒店评论示例（用于网页 Demo）
├── labels_final_2000_v4_mlp.jsonl         # 经过多轮审查与清洗确认的 2,000 条标准训练标注集
├── labels_final_2000_v4_mlp_notes.md      # 标注规范、清洗规则与审查说明
├── records_to_review.csv                  # 疑似酒店回复/需人工复核样本
│
├── server.py                              # FastAPI 后端服务（推理、单句探针、数据管理与在线重训）
├── train_mlp.py                           # PyTorch 多任务 MLP 模型架构、训练与评测脚本
├── export_to_web.py                       # 模型权重导出工具
├── load_dataset.py                        # 数据集读取示例脚本
├── get_qwen_embedding.py                  # Qwen 256 维向量生成示例
├── annotate_with_deepseek.py              # DeepSeek 辅助标注工具
│
├── index.html                             # Web 交互系统主页面
├── app.js                                 # 前端交互与 API 通信逻辑
├── style.css                              # 深色玻璃拟态样式表
│
└── experiments/                           # 核心代际模型与评测报告
    ├── run_B_default_128_64/              # Gen 1 基准模型 (100条基础数据)
    ├── run_model2_optimized_500data/      # Gen 2 优化模型 (500条数据)
    ├── run_model3_champion_1000v3/        # Gen 3 Champion 模型 (1000条数据)
    ├── run_model4_champion_2000v4/        # Gen 4 Champion 核心模型 (2000条全量数据)
    ├── run_model4_interactive_retrained/  # Web 端在线标注修正后的热重训模型
    ├── baseline_100data_report.md         # Gen 1 评测报告
    ├── model2_500data_report.md           # Gen 2 评测报告
    ├── model3_1000data_report.md          # Gen 3 评测报告
    ├── model4_2000data_report.md          # Gen 4 评测报告
    └── mlp_optimization_analysis.md       # 四代演进消融与优化全景分析
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2. 配置 API Key（可选，用于自定义文本向量化与辅助标注）

复制 `api_keys.example.txt` 为 `api_keys.txt`，并填入相应的密钥：

```bash
cp api_keys.example.txt api_keys.txt
```

或通过环境变量配置：
```bash
export DASHSCOPE_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

### 3. 启动 Web 交互系统

```bash
python server.py --host 127.0.0.1 --port 8792
```

浏览器访问：**http://127.0.0.1:8792**

- **测试登录账号**：`teacher` / `admin123`
- **主要功能页面**：
  1. **📊 酒店评论情感分析**：支持长篇多句评论分句检测、8 维度柱状图与 [-100, +100] 综合极性得分聚合。
  2. **🔍 单句探针检测**：输入任意单句，实时观察 8 个维度的概率分布与预测证据。
  3. **📈 模型指标与四代演进**：Gen 1 ~ Gen 4 演进对比表、8 维度雷达/柱图、Loss 曲线展示。
  4. **📁 训练数据可视化与修正**：浏览 2000 条训练标注集、过滤模型分歧样本、在线修正标签并一键触发后台重训与热加载！

---

## 模型训练与复现

直接运行训练脚本以复现 Gen 4 Champion 模型或重新评估：

```bash
# 使用当前 2000 条标注数据进行完整训练与评测
python train_mlp.py --data-version v4 --epochs 60 --lr 0.0008 --hidden-dims 384 192

# 仅评估当前最优模型
python train_mlp.py --eval-only
```

---

## 标注体系说明

### 八个维度 (Aspects)
1. `cleanliness`：清洁卫生、污渍、虫害等
2. `service`：员工服务态度、前台效率、行李服务等
3. `location`：地理位置、周边交通、景点距离等
4. `facilities`：硬件设施、空调、电梯、卫浴管道等
5. `room_comfort`：房间采光、床品舒适度、面积、室温等
6. `sound_insulation_noise`：隔音、走廊/街道噪音、安静程度等
7. `food`：酒店早餐与餐饮品质、口感等
8. `value`：性价比、价格、收费合理性等

### 五种状态 (States)
- `0 absent`：未提及该方面
- `1 negative`：负面评价
- `2 neutral`：中性评价（客观陈述无明显偏向）
- `3 positive`：正面评价
- `4 mixed`：同一方面在同句中兼有正反评价

---

## 评论级极性得分聚合公式

对于某一方面 $a$，提取该方面所有提及句子（$s_{ia} \in \{+1, -1, 0\}$）：

$$S_a = 100 \times \frac{\sum_{i=1}^{n_a} s_{ia}}{n_a}$$

若未提及则记为 `Not mentioned`。
