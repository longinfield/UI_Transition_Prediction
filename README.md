# [IJHCS 2025] Toward AI-driven UI transition intuitiveness inspection for smartphone apps
Dataset and Computational Models

## Benchmark Dataset
* **`Positive_data` & `Negative_data`**: Store the positive and negative samples for the binary uncertainty prediction task.
* **`UILink_test.json`**: Stores the evaluation samples for the link UI prediction task.

## Uncertainty Predictor
Given descriptions of the current page and the target page, this model computes whether users will feel uncertain about how to navigate to the target page.

---

### 🛠 Environment Setup

The code is optimized for Google Colab and requires the following python environment dependencies:
*   **Transformers**: `4.47.1`
*   **Numpy**: `1.26.4`
*   **Torch**: Mainstream PyTorch versions compatible with CUDA
*   **Other Libraries**: `openai-clip`, `scikit-learn`, `pillow`

---

### 📂 Dataset Directory Structure

To ensure that the data-loading functions (e.g., `build_train_val_set_with_hifi_data`) execute properly, please organize your dataset under the `UTP_model_training` directory in your Google Drive as follows:

```text
UTP_model_training/
├── sk.json                     # Whitelist configuration file containing specific filtering logic
├── certain_data/               # Positive/Deterministic training data (Label: 0)
│   └── [directory_name]/
│       ├── [directory_name]start.jpg
│       └── YOLO_[directory_name]end.json
├── uncertain_data/             # Negative/Uncertain/Ambiguous training data (Label: 1)
│   └── [directory_name]/
│       ├── [directory_name]start.jpg
│       ├── [directory_name]wireframe_start.jpg
│       └── hand_[directory_name]end.json
├── benchmark_data/             # Benchmark test set
│   ├── positive_data/
│   │   └── [1-140]/            # Folders named by sequential numbers
│   │       ├── *start.jpg / *start_mockup.png
│   │       └── hand_*end.json
│   └── negative_data/
│       └── [1-140]/
│           ├── *start.jpg / *start_mockup.jpg
│           └── hand_*end.json
└── candidate_models/          # Directory where model weights are automatically saved
```
---

## Link UI Predictor
This module targets the structural alignment task of UI transition analysis. It processes consecutive smartphone GUI screens to precisely identify the specific UI element (by its ID) on the prior screen that triggers the navigation to the next screen.

### 📌 Core Functionality

1.  **QLoRA Fine-Tuning Architecture**: Built upon the **LLaMA-Factory** framework to fine-tune `Meta-Llama-3.1-8B-Instruct` using 4-bit quantization, significantly reducing GPU memory footprints while retaining advanced instruction-following capabilities.
2.  **Streaming Inference & Automated Pipelines**: Implements live stream-chat inference that auto-saves responses and maps model predictions against human ground-truth labels directly into structured `.csv` sheets.
3.  **Four Prompting Experimental Strategies**: Features an encapsulated `LinkModel` class designed to benchmark UI navigation reasoning across four specialized prompting modes:
    *   **`direct_prompt`**: Baseline execution with zero external priming.
    *   **`prompt_with_knowledge`**: Prior knowledge injection featuring **five core mobile navigation principles** (Semantic Consistency, Workflow Hierarchy, Visual Variation, Common Navigation Modes, and Visual Salience).
    *   **`few_shot_without_knowledge`**: Standard 4-shot demonstrations for basic format alignment.
    *   **`few_shot_with_COT_knowledge`**: Advanced Chain-of-Thought (CoT) priming that explicitly embeds rationale-driven step-by-step logic and rule applications within the few-shot examples.

---

### 🛠 Dependencies and Hardware Requirements

The script is tailored for execution on **Google Colab (T4 or higher GPU environments recommended)**. The following setup is required:
*   **LLaMA-Factory**: `0.8.3.dev0`
*   **Transformers**: `4.45.2`
*   **Tokenizers**: `0.20.1`
*   **Bitsandbytes**: Enabled for 4-bit QLoRA operations

---

### 📂 Project Data and Configuration Standards

Ensure that the following assets are correctly mapped in the `UTP_model_training/LLaMA-Factory/` path within your Google Drive:
*   **`data/dataset_info.json`**: Registration metadata containing configuration profiles for the `UILinkII` `UILinkII_test` dataset.
*   **`data/UILinkII_test.json`**: The designated testing data.
*   **`data/UILinkII.json`**: The designated training data.
*   **`llama3.1_UILink_II_lora_epoch6/`**: The folder containing saved LoRA adapter checkpoint weights.
