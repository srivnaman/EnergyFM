<div align="center">
<img width="60%" alt="MOMENT" src="assets/MOMENT Logo.png">
<h1>MOMENT: A Family of Open Time-series Foundation Models</h1>

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2402.03885&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2402.03885)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/AutonLab/MOMENT-1-large)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/MIT)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue)]()

</div>

### MOMENT: One Model, Multiple Tasks, Datasets & Domains

<div align="center">
<img width="60%" alt="MOMENT: One Model, Multiple Tasks, Datasets & Domains" src="https://github.com/moment-timeseries-foundation-model/moment/assets/26150479/90c7d055-36d2-42aa-92b1-c5cfade22b3e">
</div>

MOMENT on different datasets and tasks, without any parameter updates:
- _Imputation:_ Better than statistical imputation baselines
- _Anomaly Detection:_ Second best $F_1$ than all baselines
- _Classification:_ More accurate than 11 / 16 compared methods
- _Short-horizon Forecasting:_ Better than ARIMA on some datasets

By linear probing (fine-tuning the final linear layer): 
- _Imputation:_ Better than baselines on 4 / 6 datasets
- _Anomaly Detection:_ Best $F_1$
- _Long-horizon Forecasting:_ Competitive in some settings

## 🧑‍💻 Usage

**Recommended Python Version:** Python 3.11 (support for additional versions is expected soon).

You can install the `momentfm` package using pip:
```bash
pip install momentfm
```
Alternatively, to install the latest version directly from the GitHub repository:
```bash
pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
```

To load the pre-trained model for one of the tasks, use one of the following code snippets:

**Forecasting**
```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        "task_name": "forecasting",
        "forecast_horizon": 96
    },
)
model.init()
```

**Classification**
```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        "task_name": "classification",
        "n_channels": 1,
        "num_class": 2
    },
)
model.init()
```

**Anomaly Detection, Imputation, and Pre-training**
```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={"task_name": "reconstruction"},
)
model.init()
```

**Representation Learning**
```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={"task_name": "embedding"},
)
model.init()
```

## 🧑‍🏫 Tutorials

<!-- We provide tutorials to demonstrate how to use and fine-tune our pre-trained model on various tasks. -->
Here is the list of tutorials and reproducibile experiments to get started with MOMENT for various tasks:
- [Forecasting](./tutorials/forecasting.ipynb)
- [Classification](./tutorials/classification.ipynb)
- [Anomaly Detection](./tutorials/anomaly_detection.ipynb)
- [Imputation](./tutorials/imputation.ipynb)
- [Representation Learning](./tutorials/representation_learning.ipynb)
- [Real-world Electrocardiogram (ECG) Case Study](./tutorials/ptbxl_classification.ipynb) -- This tutorial also shows how to fine-tune MOMENT for a real-world ECG classification problem, performing training and inference on multiple GPUs and parameter efficient fine-tuning (PEFT). 

Special thanks to [Yifu Cai](https://github.com/raycai420) and [Arjun Choudhry](https://github.com/Arjun7m) for the tutorials!
