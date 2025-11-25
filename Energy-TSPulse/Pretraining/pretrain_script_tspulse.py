import logging
import os
import tempfile
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import random
import json
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist
from torch.utils.data import IterableDataset

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse import TSPulseConfig
from tsfm_public.models.tinytimemixer.utils import get_ttm_args
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.visualization import plot_predictions

logger = logging.getLogger(__name__)


import random
import itertools
from pathlib import Path
from torch.utils.data import IterableDataset

class StreamingTimeSeriesDataset(IterableDataset):
    def __init__(
        self,
        data_root_path: str,
        context_length: int,
        prediction_length: int = 8,
        eps: float = 1e-5,
        shuffle_buffer_size: int = 100_000,
        num_epochs: int = 10,
        rank: int = 0,
        world_size: int = 1,
        base_seed: int = 42,
    ):
        self.data_root_path = data_root_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.eps = eps
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_epochs = num_epochs
        
        self.rank = rank
        self.world_size = world_size
        self.base_seed = base_seed
        
        # Find all parquet files once
        self.files = sorted(Path(data_root_path).rglob("*.parquet"))
        
    def set_epoch(self, epoch: int):
        self._epoch = epoch
    
    def parse_and_normalize(self, pq_path):
        
        df = pd.read_parquet(pq_path).reset_index()
        df = df.rename(columns={df.columns[0]: "time"})
        df_long = df.melt(id_vars="time", var_name="building_id", value_name="energy")
        df_long["time"] = pd.to_datetime(df_long["time"], errors="coerce")
        df_long = df_long.dropna(subset=["time"])
        
        for bld, g in df_long.groupby("building_id"):
            vals = g["energy"].to_numpy(dtype=np.float32)
            mask = ~np.isnan(vals)
            if not mask.any():
                continue
            
            mean = np.mean(vals[mask])
            std = np.std(vals[mask])
            std = max(std, self.eps)
            
            vals = (vals - mean) / std
            vals = np.nan_to_num(vals, nan=0.0)
            
            total_len = len(vals)
            window = self.context_length + self.prediction_length
            if total_len < window:
                continue
            
            for i in range(total_len - window + 1):
                past = vals[i : i + self.context_length]
                past_mask = mask[i : i + self.context_length]
                sample = {
                    "past_values": past,
                    "past_observed_mask": past_mask.astype(np.bool_),
                    "series_mean": mean,
                    "series_std": std,
                }
                if self.prediction_length > 0:
                    sample["future_values"] = vals[i + self.context_length : i + window]
                yield sample
    
    def _sample_iterator(self, files_to_process):
        for pq_path in files_to_process:
            yield from self.parse_and_normalize(pq_path)
    
    def __iter__(self):
        import torch.distributed as dist
        
        # Determine rank and world size for distributed training (fallback to 0,1)
        rank = self.rank
        world_size = self.world_size

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        for epoch in range(self.num_epochs):
            # Shuffle files deterministically per epoch
            files_shuffled = list(self.files)
            seed = self.base_seed + epoch
            random.Random(seed).shuffle(files_shuffled)
            
            # Split files evenly for distributed workers
            files_for_rank = list(itertools.islice(files_shuffled, rank, None, world_size))
            
            it = self._sample_iterator(files_for_rank)
            
            if self.shuffle_buffer_size <= 0:
                # immediate streaming
                yield from it
            else:
                # buffered shuffle
                buffer = []
                for _ in range(self.shuffle_buffer_size):
                    try:
                        buffer.append(next(it))
                    except StopIteration:
                        break
                random.shuffle(buffer)
                for sample in it:
                    idx = random.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = sample
                random.shuffle(buffer)
                for sample in buffer:
                    yield sample




def pad_to_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.size(1) < length:
        pad = length - x.size(1)
        x = torch.cat([x, torch.zeros(x.size(0), pad, x.size(2), device=x.device)], dim=1)
    elif x.size(1) > length:
        x = x[:, :length, :]
    return x

class TimeSeriesCollator:
    def __init__(self, context_length: int, patch_length: int, prediction_length: int = 0):
        self.context_length = context_length
        self.patch_length = patch_length
        self.prediction_length = prediction_length

    def __call__(self, features: list) -> dict:
        past = torch.stack([
            torch.nn.functional.pad(
                torch.tensor(f['past_values'], dtype=torch.float),
                (0, self.context_length - len(f['past_values'])),
                value=0.0,
            ) for f in features
        ]).unsqueeze(-1)
        past = pad_to_length(past, self.context_length)
        mask = torch.stack([
            torch.tensor(f['past_observed_mask'], dtype=torch.bool) for f in features
        ]).unsqueeze(-1)
        mask = pad_to_length(mask, self.context_length)

        batch = {'past_values': past, 'past_observed_mask': mask}
        if self.prediction_length > 0 and 'future_values' in features[0]:
            fut = torch.stack([
                torch.tensor(f['future_values'], dtype=torch.float) for f in features
            ]).unsqueeze(-1)
            batch['future_values'] = fut
        return batch




def get_tspulse_model(args, actual_num_channels):

    cfg2 = load_config("config_tsp.json")
    
    ts_cfg = TSPulseConfig(cfg2)
    ts_cfg.context_length = 512
    model = TSPulseForReconstruction(ts_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"TSPulse Model: {total_params:,} total parameters, {trainable_params:,} trainable")
    return model

def prepare_model_inputs(batch):
    model_inputs = {
        "past_values": batch["past_values"],
        "past_observed_mask": batch["past_observed_mask"]
    }
    if "future_values" in batch:
        model_inputs["future_values"] = batch["future_values"]
    return model_inputs

def pretrain(args, model, dset_train, dset_val):
    lr = args.learning_rate
    logger.info(f"Starting TSPulse dual-domain pretraining with LR: {lr}")

    ckpt_dir = os.path.join(args.save_dir, "checkpoint")
    trainer_args = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=False,
        learning_rate=lr,
        max_steps=args.max_steps,
        warmup_steps=int(args.max_steps * 0.1),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        seed=args.random_seed,
        logging_dir=os.path.join(args.save_dir, "logs"),
        logging_strategy="steps",
        logging_steps=300000,
        eval_strategy="steps" if dset_val else "no",
        eval_steps=300000,
        save_strategy="steps",
        save_steps=300000,
        save_total_limit=5,
        load_best_model_at_end=bool(dset_val),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        ignore_data_skip=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # Early stopping callback
    callbacks = []
    if args.early_stopping and dset_val:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.001
        ))

    collator = TimeSeriesCollator(
        context_length=args.context_length,
        patch_length=args.patch_length,
        prediction_length=args.forecast_length
    )

    class TSPulseDualDomainTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            mi = {
                "past_values": inputs["past_values"],
                "past_observed_mask": inputs["past_observed_mask"]
            }
            if "future_values" in inputs:
                mi["future_values"] = inputs["future_values"]
            outputs = model(**mi,return_loss=False)
            print(f"Outputs: {outputs}")
            loss = outputs.loss or torch.tensor(0.0, requires_grad=True, device=mi["past_values"].device)
            if not loss.dim():
                loss = loss.view(1)
            return (loss, outputs) if return_outputs else loss

    trainer = TSPulseDualDomainTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Resume logic
    last_ckpt = get_last_checkpoint(ckpt_dir)
    if last_ckpt:
        logger.info(f"Resuming training from checkpoint: {last_ckpt}")
        state_file = os.path.join(last_ckpt, "trainer_state.json")
        if os.path.isfile(state_file):
            logger.info("  → stripping unsupported fields from trainer_state.json")
            with open(state_file, "r") as f:
                state = json.load(f)
            # remove keys introduced after Transformers 4.40
            for key in ("best_global_step",): # Only remove keys that you are certain cause issues.
                if key in state:
                    state.pop(key)
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

    logger.info("Starting TSPulse dual-domain pretraining...")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    logger.info(f"Training completed. Final metrics: {metrics}")

    if dset_val:
        eval_metrics = trainer.evaluate(dset_val)
        trainer.log_metrics("eval", eval_metrics)
        logger.info(f"Final validation metrics: {eval_metrics}")

    save_path = os.path.join(args.save_dir, "tspulse_dual_domain_pretrained")
    trainer.save_model(save_path)
    logger.info(f"TSPulse model saved to: {save_path}")
    return save_path




def inference_and_evaluation(args, model_path, dset_test):
    model = get_model(model_path=model_path)
    
    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.random_seed,
            report_to="none",
        ),
    )
    
    logger.info("="*20 + " TSPulse Test Results " + "="*20)
    output = trainer.evaluate(dset_test)
    logger.info(f"Test metrics: {output}")
    
    predictions_dict = trainer.predict(dset_test)
    predictions_np = predictions_dict.predictions[0]
    
    if len(predictions_dict.predictions) > 1:
        backbone_embedding = predictions_dict.predictions[1]
        logger.info(f"Predictions shape: {predictions_np.shape}")
        logger.info(f"Backbone embeddings shape: {backbone_embedding.shape}")
    
    plot_path = os.path.join(args.save_dir, "plots")
    plot_predictions(
        model=trainer.model,
        dset=dset_test,
        plot_dir=plot_path,
        plot_prefix="tspulse_test_inference",
        channel=0,
    )
    logger.info(f"Plots saved to: {plot_path}")
    
    return output



def count_total_windows(data_root_path: str, context_length: int, prediction_length: int = 0) -> int:
    total = 0
    for pq_path in Path(data_root_path).rglob("*.parquet"):
        df = pd.read_parquet(pq_path).reset_index()
        df = df.rename(columns={df.columns[0]: "time"})
        df_long = df.melt(id_vars="time", var_name="building_id", value_name="energy")
        df_long["time"] = pd.to_datetime(df_long["time"], errors="coerce")
        df_long = df_long.dropna(subset=["time", "energy"])
        for _, g in df_long.groupby("building_id"):
            L = len(g)
            window = context_length + prediction_length
            total += max(0, L - window + 1)
    return total



import json
import yaml
def load_config(path):
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml or .json")


def main():
    dist.init_process_group(backend="nccl")
    print("Rank:", os.environ.get("RANK", "N/A"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A")
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    args = get_ttm_args()
    cng = load_config("config.json")
    
    # WORKING TSPulse configuration
    args.context_length = 512
    # args.forecast_length = 16
    # args.patch_length = 8
    # args.d_model = 24
    # args.num_layers = 8
    # args.dropout = 0.2
    # args.head_dropout = 0.2
    # args.decoder_d_model = args.d_model
    # args.decoder_num_layers = 2
    # args.learning_rate = 1e-4
    args.batch_size = 72
    
    # args.num_epochs = 10
    # args.early_stopping = True
    # args.num_workers = 8
    logger.info("="*50)
    logger.info("TSPulse Dual-Domain Pretraining Configuration")
    logger.info("="*50)
    logger.info(f"Context Length: {args.context_length}")
    logger.info(f"Forecast Length: {args.forecast_length}")
    logger.info(f"Patch Length: {args.patch_length}")
    logger.info(f"Patches: {args.context_length // args.patch_length}")
    logger.info(f"Model Dimension: {args.d_model}")
    logger.info(f"Encoder Layers: {args.num_layers}")
    logger.info(f"Decoder Layers: {args.decoder_num_layers}")
    logger.info(f"Dual-Domain: Enabled (FFT addition)")
    logger.info("="*50)
    
    set_seed(args.random_seed)
    # Usage inside your main():
    total_windows = 1221515699
    logger.info(f"Total sliding‐window samples: {total_windows:,}")

    # total_samples = estimate_total_samples(
    #     args.data_root_path,
    #     args.context_length,d
    #     args.forecast_length,
    # )
    # logger.info(f"Estimated total samples: {total_samples:,}")
    args.number_of_gpus = 2
    args.steps_per_epoch = total_windows // (args.batch_size * args.number_of_gpus)
    desired_epochs = 10
    args.max_steps = args.steps_per_epoch * desired_epochs

    # OPTION 1: Use with your energy dataset
    args.data_root_path = "/home/samy/Abhinav/implementations-naman/Datasets/curr_train"
    args.save_dir = "./tspulse_pretrained_energy"
    os.makedirs(args.save_dir, exist_ok=True)
    if hasattr(args, 'data_root_path') and args.data_root_path:
        logger.info(f"Loading energy datasets from: {args.data_root_path}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dset_train = StreamingTimeSeriesDataset(
            data_root_path=args.data_root_path,
            context_length=args.context_length,
            prediction_length=args.forecast_length,
            eps=1e-5,
            shuffle_buffer_size=250_000,
            num_epochs=args.num_epochs,
            rank=rank,
            world_size=world_size,
            base_seed=args.random_seed
        )
        dset_val = None
        dset_test = None

        # Use actual number of channels (1 for univariate energy data)
        actual_num_channels = 1


    # Get TSPulse model with correct channel configuratio
    model = get_tspulse_model(args, actual_num_channels)

    # Pretrain the model
    model_save_path = pretrain(args, model, dset_train, dset_val)
    logger.info("="*50)
    logger.info("TSPulse Dual-Domain Pretraining Completed!")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info("="*50)

    # Run inference and evaluation
    if dset_test is not None and len(dset_test) > 0:
        inference_and_evaluation(args, model_save_path, dset_test)
        logger.info("Inference and evaluation completed!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()