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


from tsfm_public.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction

from tsfm_public.toolkit.dataset import PretrainDFDataset, ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

warnings.filterwarnings("ignore")
SEED = 42
set_seed(SEED)


# metrics used for evaluation
def cal_cvrmse(pred, true, eps=1e-8):
    pred = np.array(pred)
    true = np.array(true)
    return np.power(np.square(pred - true).sum() / pred.shape[0], 0.5) / (true.sum() / pred.shape[0] + eps)

def cal_mae(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.mean(np.abs(pred - true))

def cal_nrmse(pred, true, eps=1e-8):
    true = np.array(true)
    pred = np.array(pred)

    M = len(true) // 24
    y_bar = np.mean(true)
    NRMSE = 100 * (1/ (y_bar+eps)) * np.sqrt((1 / (24 * M)) * np.sum((true - pred) ** 2))
    return NRMSE



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

def test(args, model, criterion, device):

    folder_path = os.path.join(args["dataset_path"], "test")
    result_path = "./predictions_new"
    
    median_res = []  
    for region in os.listdir(folder_path):

        region_path = os.path.join(folder_path, region)

        results_path = os.path.join(result_path, region)
        os.makedirs(results_path, exist_ok=True)

        res = []

        for building in os.listdir(region_path):

            building_id = building.rsplit(".csv",1)[0]

            if building.endswith('.csv'):
                file_path = os.path.join(region_path, building)
                df = pd.read_csv(file_path)
                energy_data = df['energy'].values
                dataset = TimeSeriesDataset(energy_data, args["context_length"], args["prediction_length"], args["patch_stride"])
                
                # test phase
                model.eval()
                val_losses = []
                y_true_test = []
                y_pred_test = []

                # test loop
                for x_test, y_test in tqdm(DataLoader(dataset, batch_size=1), desc=f"Testing {building_id}", leave=False):
                    x_test, y_test = x_test.unsqueeze(-1).to(device), y_test.to(device)
                    with torch.no_grad():
                        test_output = model(x_test)
                        forecast = test_output.prediction_outputs.squeeze(-1)
                        loss = criterion(forecast, y_test)
                        val_losses.append(loss.item())
                        
                        # Collect true and predicted values for RMSE calculation
                        y_true_test.extend(y_test.cpu().numpy())
                        y_pred_test.extend(forecast.cpu().numpy())
                        
                # Calculate average validation loss and RMSE
                y_true_combine = np.concatenate(y_true_test, axis=0)
                y_pred_combine = np.concatenate(y_pred_test, axis=0)
                avg_test_loss = np.mean(val_losses)
                
                y_pred_combine_unscaled = unscale_predictions(y_pred_combine, dataset.mean, dataset.std)
                y_true_combine_unscaled = unscale_predictions(y_true_combine, dataset.mean, dataset.std)
                
                # Calculate CVRMSE, NRMSE, MAE on unscaled data
                cvrmse = cal_cvrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
                nrmse = cal_nrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
                mae = cal_mae(y_pred_combine_unscaled, y_true_combine_unscaled)

                res.append([building_id, cvrmse, nrmse, mae, avg_test_loss])

        columns = ['building_ID', 'CVRMSE', 'NRMSE', 'MAE', 'Avg_Test_Loss']
        df = pd.DataFrame(res, columns=columns)
        df.to_csv("{}/{}.csv".format(results_path, 'result'), index=False)

        med_nrmse = df['NRMSE'].median()
        median_res.append([region, med_nrmse])

    med_columns = ['Dataset','NRMSE']
    median_df = pd.DataFrame(median_res, columns=med_columns)
    median_df.to_csv(f"{result_path}/median_buildings_results.csv", index=False)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--config-file', type=str, default='./config/tinyTimeMixers.json', help='Input config file path', required=True)
    file_path_arg = parser.parse_args()
    config_file = file_path_arg.config_file
    # config_file = './config/tinyTimeMixers.json'
    with open(config_file, 'r') as f:
        args = json.load(f)

    # check device 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    
    # define TTMs model
    model = model_config(args).to(device)
    model.load_state_dict(torch.load(f'{args["model_save_path"]}/best_model.pth'))

    # model's parameters
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model's parameter count is:", param)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()


    start_time = time()

    # training the model and save best parameters
    test(args=args, model=model, criterion=criterion, device=device)


    end_time = time() - start_time

    print(f"inference time taken by model is {end_time} sec")


