import os
import argparse
import gc
import json
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from transformers import AutoConfig
from safetensors.torch import load_file

# Suppress excessive logging from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import TSPulse model components and the FFT utility function
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction, get_fft

# =====================================================================================
# 1. MODEL LOADING & UTILITY FUNCTIONS
# =====================================================================================

def load_model(ckpt_path, device):
    """
    Loads a TSPulse model from a checkpoint directory.
    Handles both SafeTensors and PyTorch binary files and strips the '_orig_mod.' prefix.
    """
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
        
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    
    st_path = os.path.join(ckpt_path, "model.safetensors")
    bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
    
    if os.path.exists(st_path):
        sd = load_file(st_path, device="cpu")
    elif os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
    else:
        # If no model file, it might be a raw HF model. Try loading from_pretrained
        try:
            model = TSPulseForReconstruction.from_pretrained(ckpt_path, trust_remote_code=True).to(device)
            model.eval()
            return model, model.config.context_length
        except Exception as e:
            raise FileNotFoundError(f"No model file (model.safetensors or pytorch_model.bin) found in {ckpt_path} and failed to load with from_pretrained. Error: {e}")

    # # Strip the "_orig_mod." prefix if it exists (often added during compilation)
    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k  # k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else 
        new_sd[nk] = v
        
    model = TSPulseForReconstruction._from_config(config).to(device)
    
    # Load the state dict
    # We set strict=False because the fine-tuned model might have a different head
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model, model.config.context_length

def calc_window_metrics(y_true, y_pred, window_size):
    """
    Calculates subsequence F1, Precision, and Recall on a windowed basis.
    An anomaly in a window is detected if any point in that window is flagged.
    """
    n = len(y_true) // window_size
    TP, FP, FN = 0, 0, 0
    
    for i in range(n):
        s, e = i * window_size, (i + 1) * window_size
        true_block_has_anomaly = y_true[s:e].any()
        pred_block_has_anomaly = y_pred[s:e].any()
        
        if true_block_has_anomaly and pred_block_has_anomaly:
            TP += 1
        elif not true_block_has_anomaly and pred_block_has_anomaly:
            FP += 1
        elif true_block_has_anomaly and not pred_block_has_anomaly:
            FN += 1
            
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return prec, rec, f1

def find_best_threshold(scores, y_true, window_size):
    """
    Finds the best anomaly threshold for a building by maximizing the windowed F1 score.
    """
    best_f1, best_th = -1, 0
    
    # Use scores from normal windows to set a reasonable search range for the threshold
    # This prevents extreme outliers from skewing the threshold search space
    y_true_windows = y_true.reshape(-1, window_size)
    normal_scores = scores[y_true_windows.any(axis=1) == False]
    
    if len(normal_scores) < 2:  # Fallback if there are few or no normal windows
        min_score, max_score = np.min(scores), np.max(scores)
    else:
        min_score, max_score = np.min(normal_scores), np.max(normal_scores)

    if max_score <= min_score: # Handle case where all scores are the same
        max_score = min_score + 1e-6

    # Iterate through potential thresholds to find the best one
    for th in np.linspace(min_score, max_score, 100):
        preds = (scores > th).astype(int)
        _, _, f1 = calc_window_metrics(y_true, np.repeat(preds, window_size), window_size)
        if f1 > best_f1:
            best_f1, best_th = f1, th
            
    return best_th

# =====================================================================================
# 2. DATASET CLASS
# =====================================================================================

class AnomalyDataset(Dataset):
    """
    Dataset for creating training and evaluation samples for TSPulse.
    Handles normalization, padding, and windowing.
    """
    def __init__(self, df, context_length, training=True):
        self.context_length = context_length
        self.training = training
        self.samples = []
        self.process_data(df)

    def process_data(self, df):
        """
        Processes the dataframe by building, creating normalized, padded, and windowed samples.
        """
        df_grouped = df.groupby("building_id")
        for bid, sub_df in tqdm(df_grouped, desc="Processing data into samples", leave=False):
            x = sub_df["meter_reading"].to_numpy(dtype=np.float32)
            y = sub_df["anomaly"].to_numpy(dtype=int)
            
            # Skip series that are too short or have no variance
            if len(x) < self.context_length or np.std(x) < 1e-5:
                continue

            # Normalize the series
            mu, sd = np.mean(x), np.std(x)
            x_norm = (x - mu) / sd
            
            # Create non-overlapping windows
            for i in range(0, len(x_norm) - self.context_length + 1, self.context_length):
                window_x = x_norm[i : i + self.context_length]
                window_y = y[i : i + self.context_length]
                
                self.samples.append({
                    "past_values": torch.tensor(window_x, dtype=torch.float32).unsqueeze(-1),
                    "anomaly_labels": torch.tensor(window_y, dtype=torch.bool),
                    "building_id": bid,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # For training, create a mask to hide some values (self-supervised learning)
        # We also create a loss mask to avoid calculating loss on anomalous points
        if self.training:
            # Mask for input perturbation
            observed_mask = torch.ones(self.context_length, dtype=torch.bool)
            
            # Loss mask: don't compute loss on anomalies
            loss_mask = ~sample["anomaly_labels"]
            
            return {
                "past_values": sample["past_values"],
                "past_observed_mask": observed_mask,
                "loss_mask": loss_mask.unsqueeze(-1), # Ensure same shape as output
            }
        else:
            # For validation/testing, we don't need special masks
            return {
                "past_values": sample["past_values"],
                "anomaly_labels": sample["anomaly_labels"],
            }

# =====================================================================================
# 3. EVALUATION FUNCTION
# =====================================================================================

def evaluate_model(model, df_eval, context_length, device, fft_weight):
    """
    Evaluates a trained model on a given dataframe using the zero-shot evaluation protocol.
    """
    model.eval()
    results = []
    
    unique_bids = df_eval.building_id.unique()
    for bid in tqdm(unique_bids, desc="Evaluating model", leave=False):
        sub = df_eval[df_eval.building_id == bid]
        x = sub["meter_reading"].to_numpy(dtype=np.float32)
        y_true = sub["anomaly"].to_numpy(dtype=int)
        
        L = len(x)
        if L < context_length: continue
        
        mu, sd = x.mean(), x.std()
        if sd < 1e-5: continue
        x_norm = (x - mu) / sd
        
        # Pad the series to be a multiple of context_length
        rem = L % context_length
        pad = 0
        if rem != 0:
            pad = context_length - rem
            x_padded = np.concatenate([x_norm, np.zeros(pad, dtype=np.float32)])
            y_padded = np.concatenate([y_true, np.zeros(pad, dtype=int)])
        else:
            x_padded, y_padded = x_norm, y_true
            
        n_win = len(x_padded) // context_length
        scores = np.zeros(n_win)
        
        for i in range(n_win):
            w = x_padded[i * context_length : (i + 1) * context_length]
            t = torch.tensor(w, dtype=torch.float32, device=device).view(1, context_length, 1)
            m = torch.ones_like(t, dtype=torch.bool)
            
            with torch.no_grad():
                out = model(past_values=t, past_observed_mask=m)
                
                # Time-domain error
                rec_time = out.reconstruction_outputs.view(-1).cpu().numpy()
                score_time = np.max((rec_time - w)**2)
                
                # Frequency-domain error
                fft_true, _, _, _, _ = get_fft(t)
                fft_recon = out.reconstructed_ts_from_fft
                err_freq = (fft_recon - fft_true).pow(2)
                score_freq = torch.max(err_freq).item()
                
                scores[i] = score_time + fft_weight * score_freq

        # Find the best threshold for this building and calculate metrics
        th = find_best_threshold(scores, y_padded, context_length)
        block_preds = (scores > th).astype(int)
        y_pred = np.repeat(block_preds, context_length)
        
        # Trim padding before final metric calculation
        y_true_final = y_padded[:L]
        y_pred_final = y_pred[:L]
        
        wp, wr, wf = calc_window_metrics(y_true_final, y_pred_final, context_length)
        pp = precision_score(y_true_final, y_pred_final, zero_division=0)
        pr = recall_score(y_true_final, y_pred_final, zero_division=0)
        pf = f1_score(y_true_final, y_pred_final, zero_division=0)
        
        results.append({
            "building_id": bid, "win_precision": wp, "win_recall": wr, "win_f1": wf,
            "pt_precision": pp, "pt_recall": pr, "pt_f1": pf
        })

    return pd.DataFrame(results)

# =====================================================================================
# 4. TRAINING FUNCTION
# =====================================================================================

def train_one_split(train_df, val_df, args, output_dir, fold_num):
    """
    Fine-tunes the TSPulse model for one fold of the cross-validation.
    """
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Fold {fold_num} | Using device: {device} ---")

    # Load the base pre-trained model
    model, context_length = load_model(args.ckpt, device)
    
    # Prepare datasets
    train_dataset = AnomalyDataset(train_df, context_length, training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Optimizer based on paper's hyperparameters
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9058, 0.6616))
    
    best_val_f1 = -1
    early_stop_count = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Training")
        for batch in pbar:
            inputs = batch["past_values"].to(device)
            masks = batch["past_observed_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device) # Mask to ignore anomalies in loss

            # print(f"Inputs shape: {inputs.shape}, Masks shape: {masks.shape}, Loss mask shape: {loss_mask.shape}")
            masks = masks.unsqueeze(-1)  # Ensure masks shape matches inputs

            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(past_values=inputs, past_observed_mask=masks)
            
            # --- Dual-Domain Loss Calculation ---
            # 1. Time-domain loss (MSE on non-anomalous points)
            reconstruction = outputs.reconstruction_outputs
            squared_error_time = (reconstruction - inputs).pow(2)
            masked_error_time = squared_error_time * loss_mask
            loss_time = masked_error_time.sum() / (loss_mask.sum() + 1e-8)
            
            # 2. Frequency-domain loss
            fft_true, _, _, _, _ = get_fft(inputs)
            fft_recon = outputs.reconstructed_ts_from_fft
            squared_error_freq = (fft_recon - fft_true).pow(2)
            # We assume the loss mask applies to the frequency domain as well
            # FIX: Cast boolean mask to float before calling mean()
            masked_error_freq = squared_error_freq * loss_mask.float().mean(dim=1, keepdim=True)
            loss_freq = masked_error_freq.sum() / (loss_mask.sum() / context_length + 1e-8)

            # 3. Combined loss
            loss = loss_time + args.fft_weight * loss_freq
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation ---
        val_results_df = evaluate_model(model, val_df, context_length, device, args.fft_weight)
        avg_val_f1 = val_results_df.win_f1.mean()
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Window F1: {avg_val_f1:.4f}")
        
        # --- Early Stopping & Model Saving ---
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            early_stop_count = 0
            save_path = os.path.join(output_dir, f"best_model_fold_{fold_num}")
            model.save_pretrained(save_path)
            print(f"New best model saved with F1: {best_val_f1:.4f} to {save_path}")
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop_patience:
                print(f"Early stopping triggered after {args.early_stop_patience} epochs with no improvement.")
                break
                
    # Clean up memory
    del model, train_loader, train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return os.path.join(output_dir, f"best_model_fold_{fold_num}")


# =====================================================================================
# 5. MAIN EXECUTION
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate TSPulse for Anomaly Detection.")
    # --- Paths ---
    parser.add_argument("--csv", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the pre-trained model checkpoint directory.")
    parser.add_argument("--output_dir", type=str, default="finetuned_models", help="Directory to save fine-tuned models and results.")
    
    # --- Fine-tuning Hyperparameters (from paper) ---
    parser.add_argument("--epochs", type=int, default=20, help="Max number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=160, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.011598, help="Learning rate.")
    parser.add_argument("--early_stop_patience", type=int, default=6, help="Patience for early stopping.")
    
    # --- Evaluation Hyperparameters ---
    parser.add_argument("--fft_weight", type=float, default=0.5, help="Weight for the FFT reconstruction error in scoring and loss.")
    
    # --- System ---
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    df.sort_values(["building_id", "timestamp"], inplace=True)
    
    # Simple median imputation for any remaining NaNs
    df["meter_reading"] = df.groupby("building_id")["meter_reading"].transform(lambda s: s.fillna(s.median()))
    df.dropna(subset=['meter_reading'], inplace=True) # Drop if median is also NaN

    # --- 5-Fold Cross-Validation ---
    buildings = df['building_id'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    all_fold_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(buildings)):
        fold_num = fold + 1
        print(f"\n==================== FOLD {fold_num}/5 ====================")
        
        # Split buildings into train/val and test sets
        train_val_buildings = buildings[train_val_idx]
        test_buildings = buildings[test_idx]
        
        # Further split train_val_buildings into train and validation (e.g., 80/20 split)
        np.random.shuffle(train_val_buildings)
        val_split_idx = int(len(train_val_buildings) * 0.2)
        val_buildings = train_val_buildings[:val_split_idx]
        train_buildings = train_val_buildings[val_split_idx:]
        
        train_df = df[df['building_id'].isin(train_buildings)]
        val_df = df[df['building_id'].isin(val_buildings)]
        test_df = df[df['building_id'].isin(test_buildings)]
        
        print(f"Train buildings: {len(train_buildings)}, Val buildings: {len(val_buildings)}, Test buildings: {len(test_buildings)}")
        
        # --- Train the model for this fold ---
        best_model_path = train_one_split(train_df, val_df, args, args.output_dir, fold_num)
        
        # --- Evaluate the best model on the test set for this fold ---
        print(f"Loading best model from {best_model_path} for final evaluation on test set.")
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        final_model, context_length = load_model(best_model_path, device)
        
        fold_results_df = evaluate_model(final_model, test_df, context_length, device, args.fft_weight)
        fold_results_df['fold'] = fold_num
        all_fold_results.append(fold_results_df)
        
        avg_f1 = fold_results_df.win_f1.mean()
        print(f"Fold {fold_num} Test Window F1: {avg_f1:.4f}")
        
        del final_model
        gc.collect()
        torch.cuda.empty_cache()

    # --- Aggregate and Save Final Results ---
    final_results_df = pd.concat(all_fold_results, ignore_index=True)
    output_csv_path = os.path.join(args.output_dir, "finetune_evaluation_results_EnergyTSP.csv")
    final_results_df.to_csv(output_csv_path, index=False)
    
    print("\n==================== FINAL RESULTS ====================")
    mean_f1 = final_results_df.win_f1.mean()
    std_f1 = final_results_df.win_f1.std()
    mean_prec = final_results_df.win_precision.mean()
    std_prec = final_results_df.win_precision.std()
    mean_rec = final_results_df.win_recall.mean()
    std_rec = final_results_df.win_recall.std()
    
    print(f"Cross-validation results saved to {output_csv_path}")
    print(f"Average Window F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Average Window Precision: {mean_prec:.4f} ± {std_prec:.4f}")
    print(f"Average Window Recall: {mean_rec:.4f} ± {std_rec:.4f}")

if __name__ == "__main__":
    main()
