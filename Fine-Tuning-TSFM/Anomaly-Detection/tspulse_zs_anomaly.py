import os, gc, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoConfig
from collections import OrderedDict

# This custom import is from the modeling_tspulse.py file you are using
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction

# safetensors is a dependency of transformers, so this should be installed
from safetensors.torch import load_file


def load_model(ckpt_path, device):
    """
    Loads the model by manually cleaning the state dictionary to handle the '_orig_mod.' prefix,
    working for both .safetensors and .bin files.
    """
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)

    # Unified loading logic to find either safetensors or bin file
    safetensors_path = os.path.join(ckpt_path, "model.safetensors")
    bin_path = os.path.join(ckpt_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        print(f"[INFO] Found model.safetensors, loading weights from: {safetensors_path}")
        original_state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        print(f"[INFO] Found pytorch_model.bin, loading weights from: {bin_path}")
        original_state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {ckpt_path}")

    # Always perform the cleaning step to remove the prefix
    cleaned_state_dict = OrderedDict()
    prefix = "_orig_mod."
    for k, v in original_state_dict.items():
        if k.startswith(prefix):
            cleaned_state_dict[k[len(prefix):]] = v
        else:
            cleaned_state_dict[k] = v
            
    print("[INFO] Cleaned state dictionary keys by removing '_orig_mod.' prefix.")

    # Initialize a new model from the configuration
    model = TSPulseForReconstruction(config=config)
    
    # Load our cleaned weights into the model, ignoring any mismatches
    model.load_state_dict(cleaned_state_dict, strict=False)
    print("[INFO] Manually loaded cleaned weights into the model.")

    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model; context_length = {model.config.context_length}, patch_stride = {model.config.patch_stride}")
    return model, model.config.context_length, model.config.patch_stride

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  default="Dataset/lead/train.csv")
    p.add_argument("--ckpt", required=True, help="Checkpoint folder")
    p.add_argument("--gpu",  type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"[INFO] Device: {device}")

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    df.sort_values(["building_id","timestamp"], inplace=True)
    df["meter_reading"] = df.groupby("building_id")["meter_reading"] \
                             .transform(lambda s: s.fillna(s.median()))

    model, P, stride = load_model(args.ckpt, device)

    results = []
    for bid in tqdm(df.building_id.unique(), desc="Buildings"):
        sub = df[df.building_id==bid]
        vals = sub["meter_reading"].to_numpy(dtype=np.float32)
        lab  = sub["anomaly"].to_numpy().astype(int)
        if len(vals) < P:
            continue

        # per‐series z‑score
        μ, σ = vals.mean(), vals.std()
        if σ < 1e-5:
            continue
        scaled = (vals - μ) / σ

        all_err = np.zeros_like(scaled)
        counts  = np.zeros_like(scaled)

        # stride by your patch_stride (instead of by 1)
        for i in range(0, len(scaled) - P + 1, stride):
            x = scaled[i : i + P]
            t = torch.tensor(x, device=device).view(1, P, 1)
            m = torch.ones_like(t, dtype=torch.bool)
            with torch.no_grad():
                out = model(past_values=t, past_observed_mask=m, return_loss=False)
            
            recon = out.reconstruction_outputs
            err   = (recon - t).pow(2).view(-1).cpu().numpy()
            all_err[i : i + P] += err
            counts [i : i + P] += 1

        final = all_err / np.maximum(counts, 1)
        thresh = np.quantile(final, 0.995)
        pred   = (final > thresh).astype(int)

        p_ = precision_score(lab, pred, zero_division=0)
        r_ = recall_score(lab, pred, zero_division=0)
        f_ = f1_score(lab, pred, zero_division=0)
        results.append(dict(building_id=bid, precision=p_, recall=r_, f1=f_))

    if results:
        resdf = pd.DataFrame(results)
        print(f"\n[RESULT] Avg F1 = {resdf.f1.mean():.4f} ± {resdf.f1.std():.4f}")
        resdf.to_csv("tspulse_pretrained_NEW_1.csv", index=False)
        print("[INFO] Saved tspulse_pretrained.csv")
    else:
        print("[WARN] No long‐enough series found.")

if __name__=="__main__":
    main()