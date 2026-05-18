# [IJHCS 2025] Toward AI-driven UI transition intuitiveness inspection for smartphone apps
Dataset and Computational Models
## Uncertainty Predictor

### 核心功能
给定当前页面与目标页面的描述，计算用户是否会不确定如何才能导航到目标页面

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

---

## Link UI Predictor
### 📌 核心功能

1. QLoRA 微调架构
2. 推理与自动化评估
3. 四种 Prompt 实验策略设计
   针对移动端界面导航逻辑的复杂性，封装了 `LinkModel` 类，支持以下四种对比实验模式：
   * **`direct_prompt`（直接提示）**：无外部干预，直接输入界面元素与跳转前后文本，要求模型输出 Link ID。
   * **`prompt_with_knowledge`（先验知识注入）**：在输入中硬编码**五大移动端导航推理原则**（语义一致性、工作流层次、视觉异动、常见导航模式、视觉显著性）。
   * **`few_shot_without_knowledge`（普通少样本提示）**：提供 4 个具象的真实 UI 跳转场景作为 Few-shot 示例，引导模型对齐输出格式。
   * **`few_shot_with_COT_knowledge`（思维链 + 知识少样本提示）**：在 Few-shot 示例的 Assistant 回复中，深度融合“第几条导航原则”与具体的“推理演进步骤（CoT）”，最大化激发模型的复杂逻辑推断潜力。

---

### 🛠 依赖环境与硬件要求

代码专为 **Google Colab (建议 T4 或更高 GPU 环境)** 设计。核心依赖项包括：
* **LLaMA-Factory**: 官方前沿的大模型微调框架
* **Transformers**: `4.45.2`
* **Tokenizers**: `0.20.1`
* **Bitsandbytes**: 用于 4-bit 量化支持

脚本启动时会自动挂载 Google Drive 并切入指定工作目录安装上述依赖。

---

### 📂 项目数据与配置规范

为了让微调和评估脚本顺利执行，请确保 Google Drive 的 `UTP_model_training/LLaMA-Factory/` 路径下包含以下数据资产：
* **`data/dataset_info.json`**：需提前注册好 `UILinkII` 数据集的映射信息。
* **`data/UILinkII_test.json`**：用于模型基准推理和多方案 Prompt 实验的测试集。
* **`llama3.1_UILink_II_lora_epoch6/`**：示例adapter。
