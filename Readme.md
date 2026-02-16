# EnergyFM

EnergyFM are **pretrained time series foundation models**
applied to **energy meter data**. It provides pretrained models, example pipelines,
and tutorial notebooks for **forecasting, anomaly detection, and classification**
on large-scale smart meter datasets.

EnergyFM is designed to support:
- Zero-shot inference on unseen buildings and regions
- Fine-tuning for downstream energy analytics tasks
- Reproducible research and benchmarking

This repository accompanies the paper:

**EnergyFM: Pretrained Models for Energy Meter Data Analytics**  
*ACM e-Energy 2026*

---

## Models Included

EnergyFM currently includes two pretrained models, each designed for a distinct
class of energy time series tasks.

| Model | Primary Use | Supported Tasks |
|------|------------|-----------------|
| **Energy-TTM** | Forecasting | Short-term load forecasting |
| **Energy-TSPulse** | Representation learning | Classification, anomaly detection |

---

## 📊 Energy Benchmark Leaderboard

To compare EnergyFM with other state-of-the-art time series foundation models
for energy analytics, please visit our public benchmark leaderboard:

👉 **Energy Benchmark Leaderboard**  
https://huggingface.co/spaces/Thanvish07/EB_Leaderboard

The leaderboard provides standardized evaluations across multiple energy
forecasting, anomaly detection, and classification tasks, allowing users to
directly compare EnergyFM against other models from the literature.

We encourage users to explore the leaderboard to better understand how
EnergyFM performs relative to alternative TSFMs under consistent evaluation
settings.

---

## Repository Organization

The repository is organized **by model**, with task-specific pipelines contained
within each model directory.

```text
Dataset              # Sample from Datasets used in the orignal paper
Energy-TTM/          # Energy-TTM (forecasting)
Energy-TSPulse/      # Energy-TSPulse (classification, anomaly detection)
Notebooks/        # Tutorial notebooks and example workflows

````

---


---

## Installation

Clone the repository:

```bash
git clone https://github.com/srivnaman/EnergyFM.git
cd EnergyFM
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## Getting Started

- **Zero Shot Forecasting with Energy-TTM**  
  [Open Notebook](https://github.com/srivnaman/EnergyFM/blob/main/Notebooks/Energy-TTM-zeroshot.ipynb)

- **Zero Shot Anomaly Detection with Energy-TSPulse**  
  [Open Notebook](https://github.com/srivnaman/EnergyFM/blob/main/Notebooks/Energy-tspulse_anomaly_detection_zeroshot.ipynb)

- **Appliance Classification with Energy-TSPulse**  
  [Open Notebook](https://github.com/srivnaman/EnergyFM/blob/main/Notebooks/classification_EnergyTSPulse_finetuning.ipynb)

---

## Dependencies

EnergyFM builds on IBM’s **Granite Time Series Foundation Models (TSFM)**, which
provide the core TSFM architectures and Hugging Face integration.

* Granite TSFM GitHub: [https://github.com/ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm)
* Granite TSFM Wiki: [https://github.com/ibm-granite/granite-tsfm/wiki](https://github.com/ibm-granite/granite-tsfm/wiki)

---

## Datasets

This repository does **not** host full datasets.

* Small samples may be included for demonstration purposes
* Full datasets used for pretraining and evaluation are released separately

Links to datasets and pretrained weights are provided via Hugging Face.


---

## Issues

Please report bugs or questions via GitHub Issues:

[https://github.com/srivnaman/EnergyFM/issues](https://github.com/srivnaman/EnergyFM/issues)

---

## Citation

If you use EnergyFM in your research, please cite:

```bibtex
@article{energyfm2026,
  title   = {EnergyFM: Pretrained Models for Energy Meter Data Analytics},
  author  = {Arjunan, Pandarasamy and Srivastava, Naman and Kumar, Kajeeth
             and Jati, Arindam and Ekambaram, Vijay and Dayama, Pankaj},
  journal = {ACM e-Energy},
  year    = {2026}
}
```

---
<!-- 
## Notice

EnergyFM builds on IBM Granite TSFM models available through the Hugging Face
`transformers` ecosystem. As the project evolves, code and documentation may
change accordingly. -->


