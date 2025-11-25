# ⚡ EnergyTSPulse

EnergyTSPulse is a Time-Series Foundation Model (TSFM) supporting:

- Anomaly Detection  
- Classification  
- Pretraining 

---

##  Zero-Shot Inference

### Anomaly Detection
```
python EnergyTSPulse/AnomalyDetection/zeroshot/infer.py \
    --input data/sample.csv
```

### Classification
```
python EnergyTSPulse/Classifcation/zeroshot/infer.py \
    --input data/sample.csv
```

---

## Finetuning

### Anomaly Detection
```
python EnergyTSPulse/AnomalyDetection/finetune/train.py \
    --config EnergyTSPulse/AnomalyDetection/finetune/config.yaml
```

### Classification
```
python EnergyTSPulse/Classification/finetune/train.py \
    --config EnergyTSPulse/Classification/finetune/config.yaml
```

---

## Pretraining 

### Multi-GPU 
```
torchrun --nproc_per_node=<NUM_GPUS> \
    EnergyTSPulse/Pretraining/pretrain_script_tspulse.py \
    --config EnergyTSPulse/Pretraining/config/pretrain.yaml
```

Example:
```
torchrun --nproc_per_node=4 EnergyTSPulse/Pretraining/pretrain_script_tspulse.py \
    --config EnergyTSPulse/Pretraining/config/pretrain.yaml
```

---
