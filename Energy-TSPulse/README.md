# Energy-TSPulse

This folder contains scripts and configurations for **pretraining the TSPulse model** and preparing it for downstream energy time-series tasks.


## Pretraining TSPulse

For pretraining on multi-GPU setups, use **PyTorch Distributed Data Parallel (DDP)** via `torchrun`.

```bash
torchrun --nproc_per_node=<number_of_gpus> <file_name.py> --<config_name> <value>
```

* Replace `<number_of_gpus>` with the number of GPUs available on the node.
* Configuration arguments should match those defined in the script’s `main()` function.

## Transfer Learning and Downstream Tasks

To fully leverage the **TSPulse** architecture, it is recommended to:

1. **Pretrain TSPulse from scratch** or load a pretrained checkpoint.
2. **Restart training using transfer learning** for each downstream task.
3. Attach task-specific heads (e.g., forecasting, classification, anomaly detection) during downstream training.

This approach enables effective reuse of the pretrained backbone while supporting multiple task-specific heads.

## Notes

* Ensure all dependencies are installed before running training.
* Multi-GPU pretraining is strongly recommended for large-scale datasets.
* Check individual scripts for dataset paths, model configs, and training hyperparameters.

---
