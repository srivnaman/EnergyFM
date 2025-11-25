import os
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import tomli




from permetrics.regression import RegressionMetric

# BuildingsBench imports
from buildings_bench.models import model_factory
from buildings_bench.tokenizer import LoadQuantizer

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- Metric helpers (same as your TTM code) ----------
EPS = 1e-8

def cal_cvrmse(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    return np.sqrt(((pred - true) ** 2).sum() / pred.size) / (true.mean() + EPS)

def cal_mae(pred, true):
    return np.mean(np.abs(np.asarray(pred) - np.asarray(true)))

def cal_nrmse(pred, true, eps=1e-8):
    true = np.array(true)
    pred = np.array(pred)
    M = len(true) // 24
    y_bar = np.mean(true)
    NRMSE = 100 * (1/ (y_bar+eps)) * np.sqrt((1 / (24 * M)) * np.sum((true - pred) ** 2))
    return NRMSE

class BuildingsBenchDataset(Dataset):
    """
    Dataset adapter that converts your time series data to BuildingsBench format
    """
    
    def __init__(self, series, timestamps, ctx_len=168, pred_len=24, stride=24, 
                 building_type=0, latitude=0.0, longitude=0.0):
        
        # Handle NaN values (same as your TTM approach)
        series = series.astype(np.float32)
        if np.all(np.isnan(series)):
            self.data = None
            self.n = 0
            return

        nan_mask = np.isnan(series)
        if np.any(nan_mask):
            pd_series = pd.Series(series)
            pd_series.fillna(method='ffill', inplace=True)
            pd_series.fillna(method='bfill', inplace=True)
            series = pd_series.values

        # Store original statistics for unscaling
        self.mean = series.mean()
        self.std = series.std() + EPS
        
        # Don't normalize here - BuildingsBench handles this
        self.data = series
        
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.stride = stride
        self.building_type = building_type
        self.latitude = latitude
        self.longitude = longitude
        
        # Convert timestamps
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
        self.timestamps = timestamps
        
        # Calculate number of samples
        total_len = len(self.data)
        self.n = max(0, (total_len - ctx_len - pred_len) // stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_ctx = start_idx + self.ctx_len
        end_total = end_ctx + self.pred_len
        
        # Extract load sequence (context + prediction)
        load_sequence = self.data[start_idx:end_total]
  
        
        # Get timestamps for the FULL sequence (context + prediction)
        full_timestamps = self.timestamps[start_idx:end_total]
        full_sequence_len = self.ctx_len + self.pred_len
        
        # Create temporal features for the full sequence (normalized to [0,1])
        day_of_year = np.array([ts.dayofyear / 365.0 for ts in full_timestamps])
        hour_of_day = np.array([ts.hour / 24.0 for ts in full_timestamps])
        day_of_week = np.array([ts.dayofweek / 7.0 for ts in full_timestamps])
        
        # Create spatial features for the full sequence (normalized)
        building_type_vec = np.full(full_sequence_len, self.building_type)
        latitude_vec = np.full(full_sequence_len, self.latitude / 90.0)
        longitude_vec = np.full(full_sequence_len, self.longitude / 180.0)

        
        # BuildingsBench expects the full sequence (context + target)
        # The model will internally split it
        return {
            'load': torch.tensor(load_sequence.reshape(-1, 1), dtype=torch.float32),
            'building_type': torch.tensor(building_type_vec.reshape(-1, 1), dtype=torch.long),
            'day_of_year': torch.tensor(day_of_year.reshape(-1, 1), dtype=torch.float32),
            'hour_of_day': torch.tensor(hour_of_day.reshape(-1, 1), dtype=torch.float32),
            'day_of_week': torch.tensor(day_of_week.reshape(-1, 1), dtype=torch.float32),
            'latitude': torch.tensor(latitude_vec.reshape(-1, 1), dtype=torch.float32),
            'longitude': torch.tensor(longitude_vec.reshape(-1, 1), dtype=torch.float32),
        }

    def unscale(self, arr):
        """Denormalize predictions back to original scale"""
        return arr * self.std + self.mean

def load_buildingsbench_model(model_name, checkpoint_path, repo_path, device):
    
    # Load model configuration from TOML file
    config_path = Path(f'{repo_path}/buildings_bench/configs')
    toml_file = config_path / f'{model_name}.toml'
    
    if not toml_file.exists():
        raise ValueError(f'Config {model_name}.toml not found at {config_path}')
    
    with open(toml_file, 'rb') as f:
        toml_args = tomli.load(f)
    model_args = toml_args['model']
    
    # Create model using model_factory
    model, _, predict = model_factory(model_name, model_args)
    model = model.to(device)
    
    # Load checkpoint
    model.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    return model, predict, model_args

def setup_transforms(model, model_args, device, repo_path):
    """Setup data transforms as shown in tutorial"""
    
    transform_path = Path(os.environ.get('BUILDINGS_BENCH', '')) / 'metadata' / 'transforms'
    
    # Check if using continuous loads (Gaussian models)
    if model_args.get('continuous_loads', True):
        # Gaussian models use BoxCox transform
        # Forward transform is handled by dataset, we just need identity
        transform = lambda x: x
        
        # We'll get inverse transform from the dataset later
        inverse_transform = None
        
    else:
        # Token models use LoadQuantizer
        load_transform = LoadQuantizer(
            with_merge=True,
            num_centroids=model.vocab_size,
            device='cuda:0' if 'cuda' in device else 'cpu'
        )
        load_transform.load(transform_path)
        
        transform = load_transform.transform
        inverse_transform = load_transform.undo_transform
    
    return transform, inverse_transform

@torch.no_grad()
def evaluate_zero_shot_buildingsbench(args):
    """Main evaluation function adapted from your TTM code"""
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load BuildingsBench model
    model, predict, model_args = load_buildingsbench_model(
        args.model_name, args.checkpoint_path, args.repo_path, device
    )
    
    # Setup transforms
    transform, inverse_transform = setup_transforms(model, model_args, device, args.repo_path)
    
    test_root = Path(args.test_dataset_path)
    result_dir = Path(args.result_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    median_results = []
    # Process each location (same structure as your TTM code)
    for loc_path in sorted(p for p in test_root.iterdir() if p.is_dir()):
        loc_name = loc_path.name
        loc_nrmse_scores = []
        loc_results_path = result_dir / loc_name
        location_building_results = []
        
        parquet_files = list(loc_path.rglob("*.parquet"))
        if not parquet_files:
            print(f"[WARN] No parquet files in {loc_name}")
            continue
        
        pbar_desc = f"Location {loc_name}"
        for pq_path in tqdm(parquet_files, desc=pbar_desc, leave=False):
            try:
                df = pd.read_parquet(pq_path)
            except Exception as e:
                print(f"Could not read {pq_path}: {e}")
                continue
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            for col in df.columns:
                print(col)
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Create BuildingsBench dataset
                dataset = BuildingsBenchDataset(
                    series=df[col].values,
                    timestamps=df.index,
                    ctx_len=args.context_length,
                    pred_len=args.prediction_length,
                    stride=args.patch_stride,
                )

                # dataset.dataframe.to_csv('test_data.csv')

                
                if len(dataset) == 0:
                    continue
                
                # Create dataloader
                loader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, pin_memory=(device != 'cpu')
                )
                
                y_true, y_pred = [], []
                
                for batch in loader:
                    # Move to device
                    for k, v in batch.items():
                        batch[k] = v.to(device)
                    
                    # Split the sequence (BuildingsBench format)
                    continuous_load = batch['load'].clone()
                    continuous_targets = continuous_load[:, model.context_len:]
                    
                    # Apply forward transform
                    batch['load'] = transform(batch['load'])
                    
                    # Make prediction
                    if device == 'cuda':
                        with torch.cuda.amp.autocast():
                            predictions, distribution_params = predict(batch)
                    else:
                        predictions, distribution_params = predict(batch)

                    #print(predictions.shape, predictions)
                    #return
                    
                    # Handle inverse transform
                    if inverse_transform is None:
                        # For Gaussian models, get inverse transform from dataset
                        # This is a simplified version - you may need to adapt
                        predictions_unscaled = predictions.cpu()
                        targets_unscaled = continuous_targets.cpu()
                    else:
                        # For token models
                        predictions_unscaled = inverse_transform(predictions).cpu()
                        targets_unscaled = continuous_targets.cpu()
                    
                    y_true.append(targets_unscaled)
                    y_pred.append(predictions_unscaled)
                
                if not y_true:
                    continue
                
                # Convert to numpy and calculate metrics
                y_true_concat = torch.cat(y_true).numpy().flatten()
                y_pred_concat = torch.cat(y_pred).numpy().flatten()
                
                # Unscale using dataset statistics
                y_true_unscaled = dataset.unscale(y_true_concat)
                y_pred_unscaled = dataset.unscale(y_pred_concat)


                evaluator = RegressionMetric(y_true_unscaled.flatten(), y_pred_unscaled.flatten())
                nrmse = evaluator.NRMSE()
                mae = evaluator.MAE()

                # res = pd.DataFrame({
                #     'y_true': y_true_unscaled,
                #     'y_pred': y_pred_unscaled
                # })
                # res.to_csv(col + 'restuls.csv', index=False)
                
                # Calculate metrics (same as TTM)
                # cvrmse = cal_cvrmse(y_pred_unscaled, y_true_unscaled)
                # mae = cal_mae(y_pred_unscaled, y_true_unscaled)
                # nrmse = cal_nrmse(y_pred_unscaled, y_true_unscaled)

                # print(cvrmse, mae, nrmse)   
                
                location_building_results.append([
                    col, 
                    0,
                    nrmse, 
                    mae
                ])
    
    # Save results (same as TTM)
    if location_building_results:
        cols  = ['building_ID', 'CVRMSE', 'NRMSE', 'MAE']
        loc_df = pd.DataFrame(location_building_results, columns=cols)
        loc_df.to_csv(loc_results_path / "results.csv", index=False)

        min_nrmse = loc_df['NRMSE'].min()
        max_nrmse = loc_df['NRMSE'].max()
        avg_nrmse = loc_df['NRMSE'].mean()
        med_nrmse = loc_df['NRMSE'].median()

        median_results.append([loc_name, min_nrmse, max_nrmse, avg_nrmse, med_nrmse])
        print(f"\nLocation: {loc_name}, Median NRMSE: {med_nrmse:.4f}")



def main():
    parser = argparse.ArgumentParser(description="Zero-shot BuildingsBench inference")
    parser.add_argument("--test-dataset-path", required=True, help="Path to your test dataset")
    parser.add_argument("--model-name", default="TransformerWithGaussian-L", 
                       help="Model name (TransformerWithGaussian-S/M/L)")
    parser.add_argument("--checkpoint-path", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--repo-path", required=True, help="Path to BuildingsBench repository")
    parser.add_argument("--result-path", default="./results_buildingsbench_in", help="Results directory")
    parser.add_argument("--context-length", type=int, default=168, help="Context length")  
    parser.add_argument("--prediction-length", type=int, default=24, help="Prediction length")
    parser.add_argument("--patch-stride", type=int, default=24, help="Stride for sliding window")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    evaluate_zero_shot_buildingsbench(args)

if __name__ == "__main__":
    main()
