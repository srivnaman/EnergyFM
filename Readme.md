# EnergyFM: Pretrained Models for Energy Meter Data Analytics


We introduce EnergyFM, a set of pre-trained models specifically designed for energy meter analytics, supporting multiple downstream tasks such as energy load forecasting, anomaly detection, and classification. EnergyFM builds on IBM’s Tiny Time Mixers (TTM) and TSPulse as backbone architectures, which are lightweight, achieve state-of-the-art performance on several benchmarks, and are comparatively efficient to pre-train with modest compute resources. We adapt these architectures to the energy domain, naming them **Energy-TTM** and **Energy-TSPulse**, which are pre-trained on 1.26 billion hourly meter readings from 76,217 buildings across commercial and residential sectors, diverse building types and operational settings, and spanning multiple countries and climate zones.

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
├── EnergyTTM/
│   ├── Forecasting/
│      ├── Pretraining/
│      ├── zeroshot/
│      └── finetune/
│   
├── TSPulse/
│   ├── AnomalyDetection/
│   │   ├── zeroshot/
│   │   └── finetune/
│   │
│   └── Classification/
│       ├── zeroshot/
│       └── finetune/
│
├── UNITS/
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