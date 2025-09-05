# Kokoro-Mamba PEFT 语音克隆与有声书演播项目 - **最终交接文档**

---

## 1. 项目哲学与最终目标

我们项目的核心目标是训练一个**讲话自然的、可用于专业有声书演播的、同时支持高质量少样本语音克隆**的中文语音合成模型。

这个目标的确立，经历了从“探索 LNN/Mamba 可能性” -> “实现通用声音克隆” -> “聚焦于有声书级别的自然韵律”的演进。最终，我们确立了以下核心训练哲学：

1.  **先博学，后专精 (Two-Stage Fine-tuning)**: 我们坚信，一个优秀的语音克隆模型，不能是“一张白纸”。它必须首先在一个大规模、多说话人的通用数据集（如 AISHELL-3）上，学习什么是“自然的中文”，掌握语言的底层韵律规律。在这个“通才”的基础上，我们再用目标说话人的高质量数据（如有声书），对其进行“专才”培养，让它专注于模仿特定音色和叙事风格。
2.  **微创手术，而非器官移植 (PEFT with Mamba)**: 我们选择用现代化的 Mamba 架构，去替换原始 Kokoro 模型中陈旧的 LSTM 模块。但这并非全盘否定，而是通过参数高效微调（PEFT）技术，在保留原始模型 99% 预训练知识的前提下，为其植入一颗更强大的“韵律处理心脏”。
3.  **数据决定上限，模型逼近上限**: 我们承认，任何模型的表现，其天花板都由数据的质量和多样性决定。因此，我们的整个工作流，都围绕着如何高效、鲁棒地处理和利用数据来构建。

---

## 2. 架构总览

本项目基于 `hexgrad/Kokoro-82M` 模型。其核心改进点在于：

- **韵律建模升级**: 将核心韵律预测模块 `DurationEncoder` 内部的 `LSTM` 替换为 `Mamba` 状态空间模型，以期获得更强的长序列时序信息建模能力。
- **参数高效**: 采用 PEFT 策略，在训练中冻结了除 Mamba 模块外的所有参数，极大地降低了训练所需的计算资源和时间，并有效防止了灾难性遗忘。
- **配置文件驱动**: 所有的路径、超参数和实验设置，都由一个独立的 `config.yaml` 文件管理，实现了代码与配置的完全分离，保证了实验的可复现性和易用性。

---

## 3. 环境搭建 (WSL2)

本项目**强依赖** Linux 环境进行编译和训练。对于 Windows 用户，**必须使用 WSL2**。

1.  **安装 WSL2**: 以管理员身份打开 PowerShell, 运行 `wsl --install` 并安装 Ubuntu 发行版。
2.  **安装 NVIDIA 驱动**: 在 **Windows** 端，确保安装了最新的、支持 WSL 的 NVIDIA 显卡驱动。
3.  **克隆/迁移项目**: **将整个 `kokoro` 项目文件夹，完整地复制到 WSL 的 Linux 文件系统内部** (例如 `~/kokoro`)。**绝对不要**通过 `/mnt/` 路径从 Linux 访问 Windows 上的文件，这会导致灾难性的 I/O 性能瓶颈。
4.  **安装依赖**: 在 WSL 的项目根目录 (`~/kokoro`) 下，执行：
    ```bash
    # 建议创建一个 conda 或 venv 虚拟环境
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pyyaml tqdm transformers mamba-ssm causal-conv1d
    ```

---

## 4. 数据准备 (两阶段)

### 阶段一：通才教育数据 (AISHELL-3)

1.  **下载**: 从 [OpenSLR](http://www.openslr.org/93/) 下载 `train.tar.gz` (36G) 和 `test.tar.gz` (4.7G)。
2.  **解压**: 将两个压缩包**完整解压**到项目根目录。解压后，你的项目结构应如下：
    ```
    kokoro/
    ├── train.py
    ├── config.yaml
    └── AISHELL-3/
        ├── train/
        │   ├── content.txt
        │   └── wav/
        └── test/
            ├── content.txt
            └── wav/
    ```
3.  **配置**: 打开 `config.yaml`，确保 `paths.aishell3_unzipped_train_dir` 和 `paths.aishell3_unzipped_test_dir` 正确指向了上面解压出的 `train` 和 `test` 文件夹。

### 阶段二：专才培养数据 (你的目标声音)

1.  **录制/准备**: 准备**至少 1 小时**的、由**同一个人**朗读的、**录音质量干净**的音频（如有声书）。
2.  **切分与标注**: 将长音频切分成 3-15 秒的短句 `.wav` 文件，并为**每一个** `.wav` 创建一个同名的 `.txt` 文件，内容为对应的文本。
3.  **组织**: 将所有 `.wav` 和 `.txt` 文件放入一个新的文件夹，例如 `my_audiobook_dataset`。

---

## 5. 训练执行 (两阶段)

### **第一步：数据预处理**

在开始任何训练之前，你需要先为 AISHELL-3 数据集运行一次预处理。这个步骤会将原始数据整理成“一 `wav` 配一 `txt`”的格式。

在项目根目录下执行：
```bash
python train.py --config config.yaml --prepare-data-only
```
这个命令会根据你的 `config.yaml`，在 `processed_data` 路径下生成 `train` 和 `test` 两个文件夹，里面是整理好的数据。

### **第二步：第一阶段训练 (通才培养)**

1.  **目标**: 在 AISHELL-3 全量数据集上，训练 Mamba 模块，使其掌握通用的中文韵律。
2.  **执行**:
    ```bash
    python train.py --config config.yaml
    ```
3.  **监控**: 观察终端输出的 `loss` 和 `val_loss`。当 `val_loss` 不再显著下降时，即可认为模型已初步收敛。
4.  **产出**: 在 `output_dir` 目录中，会生成 `mamba_peft_epoch_xx.pth` 权重文件。

### **第三步：第二阶段训练 (专才培养)**

1.  **目标**: 在第一阶段的基础上，用你自己的数据集，克隆目标音色和风格。
2.  **配置**:
    *   打开 `config.yaml`。
    *   将 `paths.aishell3_unzipped_train_dir` 修改为**你自己的数据集路径** (`my_audiobook_dataset`)。
    *   将 `paths.load_mamba_from` 修改为**第一阶段训练出的 `.pth` 文件路径**。
    *   适当调低 `training.learning_rate` (例如 `5e-5`)。
3.  **执行**:
    *   **重要**: 先删除掉旧的 `processed_data` 文件夹，因为它还是 AISHELL-3 的数据。
    *   首先，为你的新数据集运行预处理：`python train.py --config config.yaml --prepare-data-only`
    *   然后，开始第二阶段训练：`python train.py --config config.yaml`

---

## 6. 推理与使用

(此部分代码尚未实现，为未来展望)

训练完成后，你将得到一个专属的 Mamba 权重。要使用它，你需要：
1.  编写一个推理脚本。
2.  加载原始的 `KModel`。
3.  将你训练好的 Mamba 权重，加载回模型的 `predictor.text_encoder.lstms[0]`。
4.  提供一段目标说话人的参考音频，用 `pipeline.audio_to_ref_s` 提取风格向量。
5.  使用 `pipeline.text_to_phonemes` 和模型的 `forward_with_tokens` 方法，传入文本和风格向量，即可生成语音。
