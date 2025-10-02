# Phoneme Detection - Advanced Two-Stage LSTM Model

This repository contains the code for  Phoneme Detection from MEG.

This model employs a specialized two-stage training regimen combined with label-averaging techniques to effectively learn from the noisy Magnetoencephalography (MEG) data.

## Model Approach

| Component | Description |
| :--- | :--- |
| **Model Architecture** | `PhonemeModelV12`: A hybrid architecture featuring an **EEG2Rep**-style 1D CNN encoder, a **Bidirectional LSTM (BiLSTM)** for temporal sequence modeling, and a **CLS Token Multi-Head Attention** pooling mechanism for final classification. |
| **Training Strategy (Stage 1)** | Initial training on **Label-Averaged Groups (K=32)** to mitigate noise and over-fitting to single-trial variance. Uses a higher learning rate ($\approx$ $3 \times 10^{-4}$). |
| **Training Strategy (Stage 2)** | Fine-tuning on a more granular level using **Multi-K Label Averaging (K=60 to 100)** with a lower learning rate ($\approx$ $2 \times 10^{-5}$) for refined feature learning. |
| **Loss Function** | **ClassBalancedFocalLoss** is used to address the class imbalance inherent in phoneme frequency. |
| **Preprocessing** | Standardized z-scoring is applied globally by the base dataset loader, with optional **per-sample z-scoring** enabled during fine-tuning for improved generalization. |
| **Inference** | Uses **Test-Time Augmentation (TTA)** with temporal shifts (`-6` to `+6` time steps) to average predictions and improve robustness. |

## Quick Start

### Prerequisites

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/LibriBrain-Phoneme-Detection.git](https://github.com/YourUsername/LibriBrain-Phoneme-Detection.git)
    cd LibriBrain-Phoneme-Detection
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data Setup:**
    Follow the official competition guidelines to download the LibriBrain data and place it in the expected directory structure (`./libribrain_data/data/`).

### Training and Submission

The primary script orchestrates the two-stage training, prediction dumping, and CSV submission generation.

```bash
# Run the complete two-stage training and submission
python src/train_lstm_v12_submit.py \
    --mode train_and_submit \
    --stage1_epochs 15 \
    --stage2_epochs 30 --stage2_lr 2e-5 \
    --multik_stage2 \
    --batch_size 16 --precision 16-mixed \
    --per_sample_zscore_train \
    --per_sample_zscore_val_avg --per_sample_zscore_val_single \
    --per_sample_zscore_holdout \
    --tta_shifts "0,2,-2,4,-4,6,-6"
