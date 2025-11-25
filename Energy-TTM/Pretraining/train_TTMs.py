import os
import random 


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from time import time
import math 
import tempfile 
import torch 
import pickle 
import logging 
import warnings
import json


import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error

import argparse


from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback
from torch.utils.data import ConcatDataset, Dataset, DataLoader


from tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction

from tsfm_public.toolkit.dataset import PretrainDFDataset, ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

warnings.filterwarnings("ignore")
SEED = 42
set_seed(SEED)



def standardize_series(series, eps=1e-8):
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std+eps)
    return standardized_series, mean, std

def unscale_predictions(predictions, mean, std, eps=1e-8):
    return predictions * (std+eps) + mean



class TimeSeriesDataset(Dataset):
    def __init__(self, data, backcast_length, forecast_length, stride=1):
        # Standardize the time series data
        self.data, self.mean, self.std = standardize_series(data)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.backcast_length - self.forecast_length) // self.stride + 1

    def __getitem__(self, index):
        start_index = index * self.stride
        x = self.data[start_index : start_index + self.backcast_length]
        y = self.data[start_index + self.backcast_length : start_index + self.backcast_length + self.forecast_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


        


def load_datasets(folder_path, backcast_length, forecast_length, stride):
    datasets = []

    for region in os.listdir(folder_path):

        region_path = os.path.join(folder_path, region)
        for building in os.listdir(region_path):

            if building.endswith('.csv'):
                file_path = os.path.join(region_path, building)
                df = pd.read_csv(file_path)
                energy_data = df['energy'].values
                dataset = TimeSeriesDataset(energy_data, backcast_length, forecast_length, stride)
                datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)
    return combined_dataset


def model_config(args):

    config = TinyTimeMixerConfig(
        context_length=args["context_length"],
        patch_length=args["patch_length"],
        num_input_channels=args["num_input_channels"],
        patch_stride=args["patch_stride"],
        d_model=args["d_model"],
        num_layers=args["num_layers"],
        expansion_factor=args["expansion_factor"],
        dropout=args["dropout"],
        head_dropout=args["head_dropout"],
        mode=args["mode"][0],
        scaling=args["scaling"],
        prediction_length=args["prediction_length"],
        is_scaling=args["is_scaling"],
        gated_attn=args["gated_attn"],
        norm_mlp=args["norm_mlp"],
        self_attn=args["self_attn"],
        self_attn_heads=args["self_attn_heads"],
        use_positional_encoding=args["use_positional_encoding"],
        positional_encoding_type=args["positional_encoding_type"],
        loss=args["loss"],
        init_std=args["init_std"],
        post_init=args["post_init"],
        norm_eps=args["norm_eps"],
        adaptive_patching_levels=args["adaptive_patching_levels"],
        resolution_prefix_tuning=args["resolution_prefix_tuning"],
        frequency_token_vocab_size=args["frequency_token_vocab_size"],
        distribution_output=args["distribution_output"],
        num_parallel_samples=args["num_parallel_samples"],
        decoder_num_layers=args["decoder_num_layers"],
        decoder_d_model=args["decoder_d_model"],
        decoder_adaptive_patching_levels=args["decoder_adaptive_patching_levels"],
        decoder_raw_residual=args["decoder_raw_residual"],
        decoder_mode=args["decoder_mode"],
        use_decoder=args["use_decoder"],
        enable_forecast_channel_mixing=args["enable_forecast_channel_mixing"],
        fcm_gated_attn=args["fcm_gated_attn"],
        fcm_context_length=args["fcm_context_length"],
        fcm_use_mixer=args["fcm_use_mixer"],
        fcm_mix_layers=args["fcm_mix_layers"],
        fcm_prepend_past=args["fcm_prepend_past"], 
        init_linear=args["init_linear"],
        init_embed=args["init_embed"],

    )

    pretraining_model = TinyTimeMixerForPrediction(config)
    return pretraining_model


def train(model, criterion, optimizer, device, train_loader, val_loader):

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    early_stop = False

    num_epochs = 100
    train_start_time = time()  # Start timer

    for epoch in range(num_epochs):

        if early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break  

        model.train()
        train_losses = []

        epoch_start_time = time()  # Start epoch timer

        # Progress bar for the training loop
        with tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for x_train, y_train in pbar:
                x_train, y_train = x_train.unsqueeze(-1).to(device), y_train.to(device)
                optimizer.zero_grad()
                output = model(x_train)
                forecast = output.prediction_outputs.squeeze(-1)
                loss = criterion(forecast, y_train)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                pbar.set_postfix(loss=loss.item(), elapsed=f"{time() - epoch_start_time:.2f}s")
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []
        y_true_val = []
        y_pred_val = []

        # Progress bar for the validation loop
        with tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for x_val, y_val in pbar:
                x_val, y_val = x_val.unsqueeze(-1).to(device), y_val.to(device)
                with torch.no_grad():
                    val_output = model(x_val)
                    forecast = val_output.prediction_outputs.squeeze(-1)
                    loss = criterion(forecast, y_val)
                    val_losses.append(loss.item())
                    
                    # Collect true and predicted values for RMSE calculation
                    y_true_val.extend(y_val.cpu().numpy())
                    y_pred_val.extend(forecast.cpu().numpy())

        # Calculate average validation loss and RMSE
        avg_val_loss = np.mean(val_losses)
        rmse_val = np.sqrt(mean_squared_error(y_true_val, y_pred_val))

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save the best model parameters
            os.makedirs(args["model_save_path"], exist_ok=True)
            torch.save(model.state_dict(), f'{args["model_save_path"]}/best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                early_stop = True


    total_training_time = time() - train_start_time
    print(f'Total Training Time: {total_training_time:.2f}s')



if __name__ == '__main__':

    # get arguments
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--config-file', type=str, default='./config/tinyTimeMixers.json', help='Input config file path', required=True)
    file_path_arg = parser.parse_args()
    config_file = file_path_arg.config_file
    # config_file = './config/tinyTimeMixers.json'
    with open(config_file, 'r') as f:
        args = json.load(f)

    train_path = os.path.join(args["dataset_path"], "train")
    val_path = os.path.join(args["dataset_path"], "val")

    # Load datasets
    train_datasets = load_datasets(train_path, args["context_length"], args["prediction_length"], args["patch_stride"])
    val_datasets = load_datasets(val_path, args["context_length"], args["prediction_length"], args["patch_stride"])

    # Create data loaders
    train_loader = DataLoader(train_datasets, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size=args["batch_size"], shuffle=True)


    # check device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define model
    model = model_config(args).to(device)

    # model's parameters
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model's parameter count is:", param)


    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training the model and save best parameters
    train(model=model, criterion=criterion, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader)
