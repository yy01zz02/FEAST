# Online DPO Training for Flux Kontext

基于 Flow Matching 的 Online DPO（Direct Preference Optimization）训练框架，用于微调 Flux Kontext 指令编辑模型。本模块实现了论文中描述的在线偏好对齐策略。

## 核心特性

- **Online DPO**: 在线采样 + 实时偏好对构建，无需预标注数据
- **Hybrid Reward**: 支持 HPS (美学) + CTR (业务) 混合奖励
- **MLLM Filter**: Qwen3-VL 质量过滤，确保生成样本质量
- **FSDP Training**: 全参数微调，支持多 GPU 分布式训练
- **EMA**: 指数移动平均，提升模型稳定性

## 环境配置

### 1. 依赖安装

```bash
# 进入项目根目录
cd FEAST

```

### 2. 模型准备

需要下载以下模型到 `data/` 目录：

```bash
mkdir -p data

# 1. Flux Kontext 基础模型
# 从 Hugging Face 下载或使用本地路径
huggingface-cli download black-forest-labs/FLUX.1-kontext

# 2. HPSv2 奖励模型 (用于美学评分)
mkdir -p data/hpsv2
# 下载 HPSv2 checkpoint 到 data/hpsv2/

# 3. Qwen3-VL-32B-Instruct (用于质量过滤，必需)
# 可以使用 API 模式或本地部署
```

### 3. 目录结构

```
FEAST/
├── data/
│   ├── flux-kontext/                    # Flux Kontext 模型
│   │   ├── transformer/
│   │   ├── vae/
│   │   └── ...
│   ├── kontext/                         # 训练数据
│   │   ├── train_metadata.jsonl         # 原始元数据
│   │   ├── ctr_metadata.csv             # CTR 元数据
│   │   └── generated_images/            # 源图像
│   ├── kontext_preprocessed/            # 预处理后的数据 (自动生成)
│   │   ├── metadata.json
│   │   ├── prompt_embed/
│   │   ├── pooled_prompt_embeds/
│   │   ├── text_ids/
│   │   └── source_latents/
│   ├── hpsv2/                           # HPSv2 模型
│   └── outputs/                         # 训练输出
│       └── kontext_dpo_hps_ctr/
├── fastvideo/
│   ├── train_dpo_flux_kontext.py        # 主训练脚本
│   ├── data_preprocess/
│   │   └── preprocess_flux_kontext_embeddings.py
│   ├── dataset/
│   │   └── latent_flux_kontext_rl_datasets.py
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── api_rewards.py               # 奖励函数
│   │   ├── server_hpsv2.py              # HPSv2 服务
│   │   └── qwen_filter_server.py        # Qwen 过滤服务
│   └── utils/
│       ├── checkpoint.py
│       ├── fsdp_util.py
│       └── ...
└── scripts/
    └── finetune/
        ├── finetune_flux_kontext_dpo_hps.sh
        ├── finetune_flux_kontext_dpo_ctr.sh
        └── finetune_flux_kontext_dpo_hps_ctr.sh
```

## 数据准备

### 1. 原始数据格式

创建 JSONL 格式的训练元数据文件 (`data/kontext/train_metadata.jsonl`)：

```jsonl
{"image_id": "10001", "prompt": "", "image": "generated_images/10001.jpg"}
{"image_id": "10002", "prompt": "", "image": "generated_images/10002.jpg"}
{"image_id": "10003", "prompt": "", "image": "generated_images/10003.jpg"}
```

**字段说明**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `image_id` | string | 图片 ID，用于关联 CTR 元数据 |
| `prompt` | string | 编辑指令 |
| `image` | string | 源图像相对路径 |

### 2. CTR 元数据格式 (必需)

创建 CSV 文件 (`data/kontext/ctr_metadata.csv`)，用于 CTR 奖励模型查询商品元信息：

```csv
image_id,food_name,food_type,city_name,shop_name
10001,红烧肉,中式快餐,上海,老王饭店
10002,牛肉面,面食,北京,兰州拉面
10003,宫保鸡丁,川菜,杭州,蜀香居
```

**注意**: `image_id` 必须与 JSONL 中的 `image_id` 字段一一对应，训练时会通过此 ID 查询对应的商品元信息。

### 3. 数据预处理

预处理会生成 T5 文本嵌入和 VAE 图像潜码：

```bash
torchrun --nproc_per_node=4 --master_port 19004 \
    fastvideo/data_preprocess/preprocess_flux_kontext_embeddings.py \
    --jsonl_path data/kontext/train_metadata.jsonl \
    --output_dir data/kontext_preprocessed \
    --model_path data/flux-kontext \
    --height 1312 \
    --width 784 \
    --dataloader_num_workers 4
```

预处理完成后，`data/kontext_preprocessed/` 目录下会生成：
- `metadata.json`: 处理后的元数据索引（包含 image_id）
- `prompt_embed/`: T5 文本嵌入
- `pooled_prompt_embeds/`: 池化文本嵌入
- `text_ids/`: 文本位置 ID
- `source_latents/`: 源图像 VAE 潜码

## 启动奖励服务

训练前需要启动相应的奖励 API 服务：

### 1. HPSv2 服务 (端口 8163)

```bash
# 设置模型路径
export HPSV2_PATH=data/hpsv2

# 启动服务
python fastvideo/rewards/server_hpsv2.py
```

### 2. CTR 服务 (端口 8199)

```bash
# 启动 CTR 预测服务 (需要先训练 CTR 模型)
cd ctr
python server.py
```

### 3. Qwen3-VL 过滤服务 (端口 8171，必需)

```bash
# 方式一：使用 OpenAI 兼容 API (推荐)
export USE_QWEN_API=true
export QWEN_API_BASE=http://your-qwen-api-server:8170/v1
export QWEN_API_KEY=your-api-key

# 方式二：本地 vLLM 部署
export USE_QWEN_API=false
export QWEN_MODEL_PATH=Qwen/Qwen3-VL-32B-Instruct

# 启动服务
python fastvideo/rewards/qwen_filter_server.py
```

## 训练模型

### 快速开始

```bash
# HPS + CTR 混合奖励训练 (推荐)
bash scripts/finetune/finetune_flux_kontext_dpo_hps_ctr.sh

# 仅 HPSv2 美学奖励
bash scripts/finetune/finetune_flux_kontext_dpo_hps.sh

# 仅 CTR 业务奖励
bash scripts/finetune/finetune_flux_kontext_dpo_ctr.sh
```

### 自定义训练

```bash
torchrun --nproc_per_node=4 --master_port 19010 \
    fastvideo/train_dpo_flux_kontext.py \
    --seed 42 \
    --pretrained_model_name_or_path data/flux-kontext \
    --data_json_path data/kontext_preprocessed/metadata.json \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 1000 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --output_dir data/outputs/kontext_dpo \
    --final_model_dir data/outputs/kontext_dpo_final \
    --h 1312 \
    --w 784 \
    --sampling_steps 16 \
    --beta 5000.0 \
    --reward_type hps_ctr \
    --ctr_metadata_csv data/kontext/ctr_metadata.csv \
    --qwen_filter_url http://localhost:8171 \
    --num_valid_samples 2 \
    --max_sample_attempts 20 \
    --use_ema \
    --ema_decay 0.995
```

### 主要参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--pretrained_model_name_or_path` | 必填 | Flux Kontext 模型路径 |
| `--data_json_path` | 必填 | 预处理后的元数据路径 |
| `--ctr_metadata_csv` | 必填 | CTR 元数据 CSV 路径（包含 image_id 对应的商品信息） |
| `--qwen_filter_url` | `http://localhost:8171` | Qwen3-VL 过滤服务 URL（必需） |
| `--train_batch_size` | 2 | 每 GPU 批次大小 |
| `--gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `--max_train_steps` | 1000 | 最大训练步数 |
| `--learning_rate` | 1e-5 | 学习率 |
| `--h` / `--w` | 1312 / 784 | 图像高度 / 宽度 |
| `--sampling_steps` | 16 | 采样步数 |
| `--beta` | 5000.0 | DPO beta 参数 (控制偏好强度) |
| `--reward_type` | `hps_ctr` | 奖励类型: `hps`, `ctr`, `hps_ctr` |
| `--num_valid_samples` | 2 | 每个 prompt 采集的有效样本数 |
| `--max_sample_attempts` | 20 | 最大采样尝试次数 |
| `--use_ema` | False | 启用 EMA |
| `--ref_update_step` | 50 | 参考模型更新频率 |

## 奖励类型说明

| 类型 | 描述 | 需要的服务 |
|------|------|------------|
| `hps` / `hpsv2` | 仅美学评分 | HPSv2 (8163) + Qwen3-VL (8171) |
| `ctr` | 仅业务 CTR | CTR (8199) + Qwen3-VL (8171) |
| `hps_ctr` | 混合奖励 (1:1) | HPSv2 (8163) + CTR (8199) + Qwen3-VL (8171) |


主要监控指标：
- `loss`: DPO 损失
- `reward_mean`: 平均奖励
- `implicit_acc`: 隐式准确率 (winner vs loser 预测准确率)
- `grad_norm`: 梯度范数

## 输出文件

训练完成后，模型保存在指定的输出目录：

```
data/outputs/kontext_dpo_hps_ctr/
├── checkpoint_100/                # 中间 checkpoint
│   └── model.safetensors
├── checkpoint_200/
├── ema_checkpoint_100/            # EMA checkpoint
│   └── ema_state_dict.safetensors
└── ...

data/outputs/kontext_dpo_hps_ctr_final/
├── model.safetensors              # 最终模型
└── ema_state_dict.safetensors     # 最终 EMA 模型
```

