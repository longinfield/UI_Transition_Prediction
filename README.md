# [IJHCS 2025] Toward AI-driven UI transition intuitiveness inspection for smartphone apps
Dataset and Computational Models
## Uncertainty Predictor
### 🛠 依赖环境设定

代码基于 Google Colab 与 PyTorch 框架开发，核心依赖版本要求如下：
* **Transformers**: `4.47.1`
* **Numpy**: `1.26.4`
* **Torch**: 兼容 CUDA 的主流 PyTorch 版本
* **其他库**: `openai-clip`, `scikit-learn`, `pillow`

可以通过代码开头的环境初始化单元直接完成安装与 Google Drive 挂载。

---

### 📂 数据目录结构规范

为了确保数据读取函数（如 `build_train_val_set_with_hifi_data`）正常运行，请将您的数据集在 Google Drive 的 `UTP_model_training` 目录下按如下结构组织：

```text
UTP_model_training/
├── sk.json                     # 包含特定筛选逻辑的白名单配置文件
├── certain_data/               # 正向/确定性数据集（Label: 0）
│   └── [directory_name]/
│       ├── [directory_name]start.jpg
│       └── YOLO_[directory_name]end.json
├── uncertain_data/             # 负向/不确定性/模糊数据集（Label: 1）
│   └── [directory_name]/
│       ├── [directory_name]start.jpg
│       ├── [directory_name]wireframe_start.jpg
│       └── hand_[directory_name]end.json
├── benchmark_data/             # 基准测试集
│   ├── positive_data/
│   │   └── [1-140]/            # 序列号命名的文件夹
│   │       ├── *start.jpg / *start_mockup.png
│   │       └── hand_*end.json
│   └── negative_data/
│       └── [1-140]/
│           ├── *start.jpg / *start_mockup.jpg
│           └── hand_*end.json
└── candidate_models/          # 模型权重自动保存目录
