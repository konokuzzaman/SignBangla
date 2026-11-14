# SignBangla — Module 1: Model Training Curves, Confusion Matrices & Evaluation

This document provides visual evidence, diagnostic curves, and comparative analysis for the deep learning models developed under **Module 1 (Skeletal Sign Language Recognition)**. 

All models are evaluated strictly under a **zero-leakage signer-independent protocol** on held-out signers **S04** and **S08**, matching the published benchmark protocol.

Live interactive runs, epoch-by-epoch loss/accuracy metrics, confusion matrices, and model checkpoints are tracked on Weights & Biases:
👉 **[W&B Project Dashboard: SignBangla-BdSLW401](https://wandb.ai/wahiduzzaman-mahin-north-south-university/SignBangla-BdSLW401)**

---

## 1. Full-Vocabulary Benchmark: BdSLW401 (401 Classes, 39,677 Trials)

### Head-to-Head Architecture Leaderboard (Signer-Independent Test Set: S04 & S08, 6,013 Samples)

| Model Architecture | Input Features | Test Top-1 Acc | Test Top-5 Acc | Macro-F1 | Weighted-F1 | Train Time (M1 MPS) | Parameters | W&B Live Run |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BiGRU + Velocity + Attention** | Coordinates + Velocities ($J + \Delta J = 402$) | **82.14%** *(4,939/6,013)* | **96.46%** | **0.7904** | **0.8077** | 44.1 mins | 664,594 | [View Run](https://wandb.ai/wahiduzzaman-mahin-north-south-university/SignBangla-BdSLW401/runs/d19wbt4q) |
| **BdSL-SPOTER v2 (Transformer)** | Coordinates + Velocities ($J + \Delta J = 402$) | **74.70%** *(4,492/6,013)* | **94.63%** | **0.7039** | **0.7264** | **24.1 mins** | 921,746 | [View Run](https://wandb.ai/wahiduzzaman-mahin-north-south-university/SignBangla-BdSLW401/runs/299adpth) |
| **BdSL-SPOTER v1 (Transformer)** | Raw Coordinates ($J = 201$) | 73.22% *(4,403/6,013)* | 93.51% | 0.6918 | 0.7165 | 27.9 mins | 896,018 | [View Run](https://wandb.ai/wahiduzzaman-mahin-north-south-university/SignBangla-BdSLW401/runs/2ty46wqv) |
| **BdSL Graph-Transformer (Phase 3)** | 4-Stream Kinematics ($J, \Delta J, B, \Delta B = 12$) + GCN | 66.19% *(3,979/6,013)* | 91.75% | 0.6135 | 0.6332 | 134.3 mins | 897,116 | [View Run](https://wandb.ai/wahiduzzaman-mahin-north-south-university/SignBangla-BdSLW401/runs/se66b907) |
| **Published Literature Baseline\*** | Raw 3D Keypoints | 75.10% | — | — | — | — | — | — |

*\*Citation: Published benchmark paper baseline for signer-independent classification on the BdSL dataset.*

---

### A. BiGRU Baseline with Velocity & Temporal Attention (Overall Accuracy Champion: 82.14%)

- **Test Top-1 Accuracy**: **82.14%** (4,939 / 6,013 correct) across all 401 classes
- **Test Top-5 Accuracy**: **96.46%**
- **Test Macro-F1**: **0.7904** | **Weighted-F1**: **0.8077**
- **Test Macro-Precision**: **0.8230** | **Macro-Recall**: **0.8091**
- **Architecture**: 2-layer Bidirectional GRU (hidden dimension 128, dropout 0.2) + learned Temporal Attention pooling + LayerNorm + Linear head.
- **Reference Comparison**: Outperforms the published literature baseline of 75.1% by **+7.04% absolute margin**.

#### Training & Validation Curves (30 Epochs)
![BiGRU 401 Training Curves](figures/bigru_401_training_curves.png)

*Figure 1: BiGRU Training vs. Validation Loss, Top-1/Top-5 Accuracy, and Macro-F1 over 30 epochs on BdSLW401. Best validation checkpoint saved at epoch 28 (Val Macro-F1: 0.7497).*

#### Signer-Independent 401×401 Confusion Matrix
![BiGRU 401 Confusion Matrix](figures/bigru_401_confusion_matrix.png)

*Figure 2: Row-normalized 401×401 Confusion Matrix on held-out signers S04 and S08. The bright, sharp diagonal demonstrates high classification precision across the entire 401-word vocabulary.*

#### Top Confused Sign Pairs (BiGRU)
| True Sign | Predicted As | Misclassified Count | Error Rate on Class | Key Reason |
|:---|:---|:---:|:---:|:---|
| **W086** | W163 | 14 | 70.0% | Symmetrical circular hand movement in front of torso |
| **W235** | W305 | 13 | 76.5% | Similar wrist angle and finger tucking gesture |
| **W105** | W361 | 12 | 63.2% | Head-level tapping gesture with identical dominant hand |
| **W090** | W005 | 10 | 55.6% | Close palm orientation and forward extension |
| **W104** | W158 | 10 | 50.0% | Subtle knuckle bending difference |

---

### B. BdSL-SPOTER v2: Kinematic Transformer (Fastest Convergence Champion: 74.70%)

- **Test Top-1 Accuracy**: **74.70%** (4,492 / 6,013 correct) across all 401 classes
- **Test Top-5 Accuracy**: **94.63%**
- **Test Macro-F1**: **0.7039** | **Weighted-F1**: **0.7264**
- **Test Macro-Precision**: **0.7477** | **Macro-Recall**: **0.7289**
- **Architecture**: 4-layer Transformer Encoder (d_model=128, 4 heads, GELU, Pre-LN) + Sinusoidal Positional Encoding + Temporal Attention Pooling.
- **Training Efficiency**: Trained in **24.1 minutes** on Apple Silicon M1 (MPS), converging in 30 epochs with 3-epoch warmup and label smoothing ($0.1$).

#### Training & Validation Curves (30 Epochs)
![BdSL-SPOTER v2 Training Curves](figures/spoter_v2_401_training_curves.png)

*Figure 3: BdSL-SPOTER v2 training progression. Label smoothing successfully bounded validation loss divergence and stabilized generalization.*

#### Signer-Independent 401×401 Confusion Matrix
![BdSL-SPOTER v2 Confusion Matrix](figures/spoter_v2_401_confusion_matrix.png)

*Figure 4: Row-normalized 401×401 Confusion Matrix for BdSL-SPOTER v2.*

---

### C. BdSL Graph-Transformer: Phase 3 SOTA Exploration (66.19%)

- **Test Top-1 Accuracy**: **66.19%** (3,979 / 6,013 correct)
- **Test Top-5 Accuracy**: **91.75%**
- **Test Macro-F1**: **0.6135** | **Weighted-F1**: **0.6332**
- **Architecture**: 4-stream Kinematics ($J, \Delta J, B, \Delta B$) + Spatial Graph Convolution over physical and adaptive skeleton graph + Dynamic Joint-Attention Pooling + 4-Layer Temporal Transformer.

#### Training & Validation Curves (30 Epochs)
![Graph-Transformer Training Curves](figures/graph_transformer_401_training_curves.png)

*Figure 5: Graph-Transformer loss and accuracy progression across 30 epochs.*

#### Signer-Independent 401×401 Confusion Matrix
![Graph-Transformer Confusion Matrix](figures/graph_transformer_401_confusion_matrix.png)

*Figure 6: Row-normalized 401×401 Confusion Matrix for the Spatial Graph-Transformer.*

---

### D. Architectural Findings & Empirical Analysis

1. **Sequential Continuity vs. Global Self-Attention**:
   - BiGRU outperformed Transformer by **+7.44%** on Top-1 accuracy (82.14% vs 74.70%).
   - Continuous sign gestures have strong frame-to-frame Markovian temporal continuity. BiGRU's recurrent hidden state naturally models smooth sequential transitions without requiring quadratic sequence self-attention.
2. **The "GCN Over-Smoothing" Bottleneck**:
   - The Spatial Graph Convolution model achieved 66.19% vs 74.70% for pure Transformer and 82.14% for BiGRU.
   - In sign language, 42 of the 67 joints belong to dense hand landmarks. Local graph aggregation smooths out fine millimeter-scale fingertip nuances, while rigid physical skeletal edges penalize free-floating hand-to-face and two-hand interactions.
3. **Impact of Dynamic Joint Velocity**:
   - Incorporating 1st-order temporal differences $\Delta(x, y, z)$ directly improved SPOTER Top-1 accuracy from **73.22% to 74.70%** and Top-5 from **93.51% to 94.63%**, proving that explicit motion vectors provide scale-invariant signing velocity cues.

---

## 2. Phase 1 Benchmark: BdSLW60 (60 Classes, 8,285 Trials)

For historical continuity and ablation comparison, the models were also trained on the 60-sign subset:

| Model | Test Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | Trainable Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BiGRU + Attention (60 Class)** | **86.79%** | **0.8353** | **0.8590** | **0.8606** | **0.8429** | 1.85 M |
| **BiLSTM + Attention (60 Class)** | 86.04% | 0.8293 | 0.8468 | 0.8563 | 0.8421 | 2.47 M |
| **BdSLW60 Benchmark Paper\*** | 75.10% | — | — | — | — | — |

### Training Curves & Confusion Matrices (60 Class)
- **BiGRU Curves**: `figures/bigru_training_curves.png`
- **BiGRU Confusion Matrix**: `figures/bigru_confusion_matrix.png`
- **BiLSTM Curves**: `figures/bilstm_training_curves.png`
- **BiLSTM Confusion Matrix**: `figures/bilstm_confusion_matrix.png`
