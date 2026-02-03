# CTR Reward Model

多模态 CTR（点击率）预测模型，用于评估食品图片的商业价值。本模块是 FEAST 框架中的核心组件，为 Online DPO 训练提供业务导向的奖励信号。

## 模型架构

该模型融合视觉和文本特征进行 CTR 预测：

- **文本编码器**: Chinese-BERT-wwm-ext
- **图像编码器**: ConvNeXt V2 Huge (ImageNet-22K 预训练)
- **跨模态融合**: Cross-Modal Transformer


## 环境配置

### 1. 依赖安装

```bash
# 使用 uv (推荐)
uv pip install torch torchvision transformers pillow pandas numpy fastapi uvicorn

# 或使用 pip
pip install torch torchvision transformers pillow pandas numpy fastapi uvicorn
```

### 2. 预训练模型下载

需要下载以下预训练模型到本地目录：

```bash
# 使用 huggingface-cli 下载
huggingface-cli download hfl/chinese-bert-wwm-ext 
huggingface-cli download facebook/convnextv2-huge-22k-512 
```

## 数据准备

### 训练数据格式

训练数据需要准备为 CSV 格式，包含以下字段：

| 字段名 | 类型 | 描述 | 示例 |
|--------|------|------|------|
| `image_url` | string | 图片本地路径 | `./images/10001.jpg` |
| `food_name` | string | 菜品名称 | `红烧肉饭` |
| `food_type` | string | 菜品类型 | `中式快餐` |
| `city_name` | string | 城市名称 | `上海` |
| `shop_name` | string | 店铺名称 | `老王饭店` |
| `pv_ctr` | float | 点击率标签 | `0.0523` |

**CSV 示例** (`dataset/train_seed_42_split_80.csv`):

```csv
image_url,food_name,food_type,city_name,shop_name,pv_ctr
./images/10001.jpg,红烧肉饭,中式快餐,上海,老王饭店,0.0523
./images/10002.jpg,鱼香肉丝,川菜,北京,川味小馆,0.0412
./images/10003.jpg,宫保鸡丁,川菜,杭州,蜀香居,0.0678
```

### 目录结构

```
ctr/
├── dataset/
│   ├── train_seed_42_split_80.csv    # 训练集
│   ├── test_seed_42_split_20.csv     # 测试集
│   └── images/                       # 图片目录
│       ├── 10001.jpg
│       ├── 10002.jpg
│       └── ...
├── output/                           # 输出目录 (自动创建)
│   └── seed_42_split_80/
│       ├── best_model.pt
│       └── checkpoint_epoch_*.pt
├── dataset.py                        # 数据集类
├── model.py                          # 模型定义
├── train.py                          # 训练脚本
├── predict.py                        # 推理类
├── server.py                         # API 服务
└── run.sh                            # 训练启动脚本
```

## 训练模型

### 快速开始

```bash
# 直接运行训练脚本
bash run.sh
```

### 自定义训练参数

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --train-csv ./dataset/train.csv \
    --test-csv ./dataset/test.csv \
    --output-dir ./output/experiment_1 \
    --bert-model models--hfl--chinese-bert-wwm-ext \
    --image-model models--facebook--convnextv2-huge-22k-512 \
    --target-image-size 512 \
    --batch-size 16 \
    --epochs 10 \
    --lr 1e-3 \
    --lr-bert 2e-5 \
    --lr-image 5e-6 \
    --use-amp \
    --accumulation-steps 4
```

### 主要参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--train-csv` | 必填 | 训练集 CSV 路径 |
| `--test-csv` | 必填 | 测试集 CSV 路径 |
| `--output-dir` | `./output` | 模型输出目录 |
| `--batch-size` | 16 | 批次大小 |
| `--epochs` | 10 | 训练轮数 |
| `--lr` | 1e-3 | 主学习率 (投影层 + 融合层) |
| `--lr-bert` | 2e-5 | BERT 学习率 |
| `--lr-image` | 5e-6 | 图像编码器学习率 |
| `--target-image-size` | 512 | 图像输入尺寸 |
| `--use-amp` | False | 启用混合精度训练 |
| `--accumulation-steps` | 4 | 梯度累积步数 |
| `--freeze-bert` | False | 冻结 BERT 参数 |
| `--freeze-image` | False | 冻结图像编码器参数 |

## 部署 API 服务

### 启动服务

```bash
# 修改 server.py 中的 checkpoint_path 为实际路径
python server.py
```

服务将在 `http://127.0.0.1:8199` 启动。

### API 调用示例

```bash
# 使用 curl 调用
curl -X POST "http://127.0.0.1:8199/predict" \
    -F "food_name=红烧肉饭" \
    -F "food_type=中式快餐" \
    -F "city_name=上海" \
    -F "shop_name=老王饭店" \
    -F "image=@./test_image.jpg"
```

**响应示例**:

```json
{
    "success": true,
    "data": {
        "ctr": 0.052341,
        "ctr_percent": "5.23%"
    }
}
```


## 使用推理类

```python
from predict import CTRPredictor

# 初始化预测器
predictor = CTRPredictor(
    checkpoint_path='./output/seed_42_split_80/best_model.pt',
    bert_model_name="models--hfl--chinese-bert-wwm-ext",
    image_model_name="models--facebook--convnextv2-huge-22k-512"
)

# 预测
ctr = predictor.predict(
    food_name="红烧肉饭",
    food_type="中式快餐",
    city_name="上海",
    shop_name="老王饭店",
    image_path="./images/10001.jpg"
)
print(f"预测 CTR: {ctr:.4f}")
```





