import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

# Make sure this path is correct for your project structure
from tsfm_public.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- Metric Helpers ----------
EPS = 1e-8
def cal_cvrmse(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    mean_true = true.mean()
    if abs(mean_true) < EPS: return np.inf
    return np.sqrt(np.mean((pred - true) ** 2)) / mean_true

def cal_mae(pred, true):
    return np.mean(np.abs(np.asarray(pred) - np.asarray(true)))

def cal_nrmse(pred, true, eps=1e-8):
    true = np.array(true)
    pred = np.array(pred)

    if len(true) < 24: return np.nan
    M = len(true) // 24
    if M == 0: return np.nan
    
    y_bar = np.mean(true)
    if abs(y_bar) < eps: return np.inf
    
    return 100 * (1 / y_bar) * np.sqrt((1 / (24 * M)) * np.sum((true - pred) ** 2))

# ---------- Dataset Class ----------
class ZeroShotDataset(Dataset):
    def __init__(self, series, ctx_len, pred_len, stride):
        series = series.astype(np.float32)
        
        if np.all(np.isnan(series)):
            self.data = None
            self.n = 0
            return

        # Using zero-imputation as per your latest version
        nan_mask = np.isnan(series)
        if np.any(nan_mask):
            series[nan_mask] = 0.0

        self.mean = series.mean()
        self.std  = series.std() + EPS
        self.data = (series - self.mean) / self.std
        
        self.ctx, self.pred, self.stride = ctx_len, pred_len, stride
        num_samples = (len(self.data) - ctx_len - pred_len) // stride + 1
        self.n = max(0, num_samples)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.stride
        x = self.data[s : s + self.ctx]
        y = self.data[s + self.ctx : s + self.ctx + self.pred]
        
        return (torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(y, dtype=torch.float32).unsqueeze(-1))

    def unscale(self, arr):
        return arr * self.std + self.mean

# ---------- Model Builder ----------
def build_model(cfg):
    model_cfg = TinyTimeMixerConfig(**cfg)
    return TinyTimeMixerForPrediction(model_cfg)

@torch.no_grad()
def evaluate_zero_shot(cfg, model, criterion, device):
    test_root   = Path(cfg["test_dataset_path"])
    result_dir  = Path(cfg["result_path"])
    result_dir.mkdir(parents=True, exist_ok=True)

    median_results = []

    for loc_path in tqdm(sorted(p for p in test_root.iterdir() if p.is_dir()), desc="Processing Locations"):
        loc_name = loc_path.name
        loc_results_path = result_dir / loc_name
        location_building_results = []

        parquet_files = list(loc_path.rglob("*.parquet"))
        if not parquet_files:
            print(f"[WARN] No .parquet files found in {loc_name}. Skipping.")
            continue
        
        loc_results_path.mkdir(exist_ok=True)

        for file_path in tqdm(parquet_files, desc=f"  Location {loc_name}", leave=False):
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"\nCould not read {file_path}: {e}")
                continue

            for building_id in df.columns:
                if not pd.api.types.is_numeric_dtype(df[building_id]):
                    continue

                series = df[building_id].values
                if len(series) < cfg["context_length"] + cfg["prediction_length"]:
                    continue

                dataset = ZeroShotDataset(
                    series=series,
                    ctx_len=cfg["context_length"],
                    pred_len=cfg["prediction_length"],
                    stride=cfg["patch_stride"],
                )
                if len(dataset) == 0: continue

                model.eval()
                val_losses, y_true_test, y_pred_test = [], [], []

                for x_test, y_test in tqdm(DataLoader(dataset, batch_size=1), desc=f"Testing {building_id}", leave=False):
                    # --- FIX: Removed the extra .unsqueeze(-1) on x_test ---
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    
                    # Using with torch.no_grad() context manager is cleaner
                    # autocast is still kept as it was in your original code
                    with autocast(device_type=device.split(':')[0], dtype=torch.bfloat16, enabled=(device != 'cpu')):
                        test_output = model(x_test)
                        forecast = test_output.prediction_outputs # Shape is (1, pred_len, 1)
                        loss = criterion(forecast, y_test)
                        val_losses.append(loss.item())
                        
                    y_true_test.append(y_test.cpu().numpy())
                    y_pred_test.append(forecast.cpu().numpy())
                        
                if not y_true_test: continue

                y_true = np.concatenate(y_true_test, axis=0).squeeze(-1)
                y_pred = np.concatenate(y_pred_test, axis=0).squeeze(-1)
                avg_test_loss = np.mean(val_losses)
                
                y_pred_unscaled = dataset.unscale(y_pred)
                y_true_unscaled = dataset.unscale(y_true)
                
                cvrmse = cal_cvrmse(y_pred_unscaled, y_true_unscaled)
                nrmse = cal_nrmse(y_pred_unscaled, y_true_unscaled)
                mae = cal_mae(y_pred_unscaled, y_true_unscaled)
                
                location_building_results.append([building_id, cvrmse, nrmse, mae, avg_test_loss])
        
        if location_building_results:
            cols = ['building_ID', 'CVRMSE', 'NRMSE', 'MAE', 'Avg_Test_Loss']
            loc_df = pd.DataFrame(location_building_results, columns=cols)
            loc_df.to_csv(loc_results_path / "results.csv", index=False)
            med_nrmse = loc_df['NRMSE'].median()
            median_results.append([loc_name, med_nrmse])

    med_cols = ['Dataset', 'NRMSE']
    median_df = pd.DataFrame(median_results, columns=med_cols)
    median_df.to_csv(result_dir / "median_results.csv", index=False)
    print(f"\n✅ Evaluation complete. Results saved to {result_dir}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Tiny Time Mixer inference")
    parser.add_argument("--config-file", required=True, help="Path to the pre-training JSON config file.")
    parser.add_argument("--test-dataset-path", required=True, help="Path to the root directory of the test dataset.")
    parser.add_argument("--result-path", default="./hopeless", help="Path to save evaluation results.")
    args = parser.parse_args()

    with open(args.config_file) as f:
        cfg = json.load(f)
    
    cfg["test_dataset_path"] = args.test_dataset_path
    cfg["result_path"] = args.result_path

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = build_model(cfg).to(device)

    if "model_save_path" not in cfg:
        raise KeyError("'model_save_path' not found in the provided config file.")

    ckpt_path = Path(cfg["model_save_path"]) / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Expected 'best.pth'.")

    print(f"Loading weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded successfully ({num_params:,} trainable parameters)")

    evaluate_zero_shot(cfg, model, criterion=nn.HuberLoss(), device=device)

if __name__ == "__main__":
    main()