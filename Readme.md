# EnergyFM: Pretrained Models for Energy Meter Data Analytics

This repository provides a unified implementation of **Time Series Foundation Models (TSFMs)**, including:

- **EnergyTSPulse**
- **TSPulse**
- **UNITS**
- **MOMENT**
- *(extendable to others)*

Each model family supports:
- **Forecasting**
- **Anomaly Detection**
- **Classification**

And each task supports two operational modes:
- **Zero-shot Inference**
- **Finetuning**

This consistent structure enables modular, scalable experimentation across TSFM architectures.

---

### Python Version Support

The following Python versions are supported:
**Python 3.10**  **Python 3.11** **Python 3.12**

---

# Getting Started


We separate the implementations of each Time-Series Foundation Model (TSFM) into individual folders.
Inside each TSFM folder, the code is further divided into downstream tasks: Forecasting, Anomaly Detection and Classification

Each task contains:

zeroshot/ → zero-shot evaluation

finetune/ → task-specific training

pretrained/ (optional) → pretrained checkpoints

Please choose the TSFM model you want to work with (e.g., EnergyTSPulse, UNITS, MOMENT, TimesFM, etc.).



##  Installation

### Clone the repository  
```bash
git clone "https://github.com/ibm-granite/granite-tsfm.git" 
cd granite-tsfm
```

### Create Virtual Environment 
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate      # Windows
```
### Install Required Dependancies
```bash
pip install ".[notebooks]"
```

## Directory Structure
```
.
├── EnergyTSPulse/
│   ├── AnomalyDetection/
│   │   ├── zeroshot/
│   │   └── finetune/
│   │
│   └── Classification/
│       ├── zeroshot/
│       └── finetune/
│   
├── TSPulse/
│   ├── Forecasting/
│   │   ├── zeroshot/
│   │   └── finetune/
│   │
│   ├── AnomalyDetection/
│   │   ├── zeroshot/
│   │   └── finetune/
│   │
│   └── Classification/
│       ├── zeroshot/
│       └── finetune/
│
├── UNITS/
│   ├── Forecasting/
│   │   ├── zeroshot/
│   │   └── finetune/
│   │
│   ├── AnomalyDetection/
│   │   ├── zeroshot/
│   │   └── finetune/
│   │
│   └── Classification/
│       ├── zeroshot/
│       └── finetune/
│
├── MOMENT/
│   ├── Forecasting/
│   │   ├── zeroshot/
│   │   └── finetune/
│   │
│   ├── AnomalyDetection/
│   │   ├── zeroshot/
│   │   └── finetune/
│   │
│   └── Classification/
│       ├── zeroshot/
│       └── finetune/
│
└── README.md

```
