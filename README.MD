# 🐶 宠物拆家行为识别项目

本项目基于 MoViNet（Mobile Video Networks）模型，通过视频识别宠物是否在进行“拆家”行为（如抓咬沙发、咬鞋、翻倒物品等）。适用于家庭监控、宠物看护、智能告警等场景。

---

## 📌 项目特色

* 🚀 使用 Google MoViNet 模型，轻量级、低延迟，适合边端设备部署
* 🐱🐶 识别宠物是否在“拆家”
* 🎥 支持任意长度视频推理（定长固定窗口处理）
* 📦 提供完整训练和推理脚本
* 📊 输出识别类别与置信度

---

## 🧠 MoViNet 简介

MoViNet 是 Google 提出的轻量级视频动作识别网络，具备以下特点：

* ✔️ 可因果卷积（适合流式输入）
* ✔️ 时间建模 + 空间建模 同步处理
* ✔️ 端到端训练，适配视频序列
* ✔️ 多种规格（A0 \~ A5），可按需选择

本项目使用 MoViNet-A0 （stream 或 base 模式）作为主干网络，并进行了二分类微调（拆家 / 正常）。

---

## 📁 数据集准备

本项目支持自行采集或公开数据集构建，建议组织如下：

```
data/
├── mess/        # 拆家视频
├── normal/      # 正常行为
```

每段视频建议 5~~10 秒，帧率 25~~30fps，命名自由，格式支持 `.mp4/.avi/.mov` 等。

### ✅ 数据划分建议

| 类型     | 示例数量 | 建议用途    |
| ------ | ---- | ------- |
| mess   | 500+ | 训练 + 验证 |
| normal | 500+ | 训练 + 验证 |

划分方式：

* 训练集：80%
* 验证集：20%
* 可用脚本随机划分或使用 sklearn `train_test_split`

---

## ⚙️ 安装依赖环境

```bash
conda create -n movinet python=3.9
conda activate movinet

pip install torch torchvision
tpip install einops opencv-python tqdm
pip install tensorflow tensorflow-hub
```

---

## 🏋️‍♂️ 模型训练

```bash
python train.py \
  --data_root ./data \
  --epochs 20 \
  --batch_size 4 \
  --num_classes 2 \
  --checkpoint_dir ./checkpoints
```

默认使用 32 帧 + 图像尺寸 172×172 (适配 MoViNet-A0)。

---

## 🔍 模型推理

推理脚本支持对任意视频段进行识别：

```bash
python infer.py \
  --video ./test_video.mp4 \
  --weights ./checkpoints/best_model.pth
```

输出示例：

```
[预测结果] 类别: mess（拆家） | 信心度: 0.91
```

---

## 💡 推理逻辑说明

1. 使用 OpenCV 读取视频并提取固定数量帧（默认 32 帧）
2. 每帧 resize 为 172x172 并归一化处理
3. 堆叠为 `[B, C, T, H, W]` 格式输入模型
4. 输出分类结果（拆家 / 正常）和对应信心度

---

## 📊 模型评估（可选）

训练阶段可输出准确率 / 混淆矩阵：

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=['normal', 'mess']))
```

---

## 📦 模型导出（ONNX / TorchScript）

可用于部署边端设备：

```python
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
```

---

## 🚧 TODO

* [x] MoViNet 模型训练与推理流程
* [x] 视频拆帧预处理支持
* [ ] 流式推理（逐帧）
* [ ] Streamlit Web UI 展示
* [ ] Android / Jetson 移动部署适配

---

## 📌 目录结构说明

```
.
├── data/                  # 数据目录
│   ├── mess/
│   └── normal/
├── movinets/              # MoViNet 模型定义
├── train.py               # 模型训练入口
├── infer.py               # 模型推理入口
├── dataset.py             # 视频数据加载器
├── checkpoints/           # 训练输出模型
├── utils.py               # 工具函数（预处理等）
└── README.md
```

---

## 📌 参考资料

* [MoViNet 原论文 (Google)](https://arxiv.org/abs/2103.11511)
* [TF-Hub MoViNet 模型](https://tfhub.dev/s?deployment-format=lite&q=movinet)
* [PyTorch MoViNet](https://github.com/tensorflow/models/tree/master/official/projects/movinet)

---

## 👤 作者信息

* 作者：Kend（中国 🇨🇳）
* 邮箱：[kend@example.com](mailto:kend@example.com)
* 时间：2025年7月

如有交流或部署协作需求，请通过邮件联系我。
