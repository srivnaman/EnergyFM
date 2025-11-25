import os
import glob
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from momentfm import MOMENTPipeline

from torch.utils.data import Dataset, DataLoader



class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.var_ = None
    
    def fit(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.var_ = np.var(data, axis=0)
    
    def transform(self, data):
        return (data - self.mean_) / np.sqrt(self.var_)
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, scaled_data):
        return (scaled_data * np.sqrt(self.var_)) + self.mean_

def make_dataframe(files):
    dataframes = []
    for file in files:
        df = pd.read_parquet(file)
        df.reset_index(inplace=True, drop=True)
        dataframes.append(df)

    merge_df = pd.concat(dataframes, axis=1)
    return merge_df


class TSDataset(Dataset):
    def __init__(
        self,
        forecast_horizon: int = 24,
        data_stride_len: int = 24,
        task_name: str = "forecasting",
        random_seed: int = 42,
        data: str = None,
    ):
        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed
        self.data = data
        self._read_data()

    def _read_data(self):
        self.scaler = CustomStandardScaler()
        np_data = self.data.values

        try:
            # handle nan values
            last_non_nan_index = np.where(~np.isnan(np_data))[0]
            last_index = last_non_nan_index[-1]
            self.np_data = np_data[:last_index + 1].reshape(-1,1)
        except Exception as e:
            print(f"Error processing data: {e}")
            self.np_data = np_data.reshape(-1, 1)


        # self.scaler.fit_transform(self.np_data)
        self.length_timeseries = len(self.np_data)


    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        timeseries = self.np_data[seq_start:seq_end, :].T
        forecast = self.np_data[seq_end:pred_end, :].T

        return timeseries, forecast, input_mask

    def __len__(self):
        return (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1
    

def create_building_loaders(file_path, batch_size=8):
    df = make_dataframe(file_path)
    building_ids = df.columns
    test_loaders = {}
    
    for building_id in building_ids:
        data = df[building_id]
        test_dataset = TSDataset(
            random_seed=13, 
            forecast_horizon=24, 
            data = data
        )
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders[building_id] = test_loader
    return test_loaders



def forecasting(files, dataset):

    test_loaders = create_building_loaders(files)
    # save_filename = os.path.basename(file).split(".")[0]
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': 24,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True, # Freeze the patch embedding layer
            'freeze_embedder': True, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        },
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )
    model.init()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)

    trues, preds, histories, losses = [], [], [], []
    model.eval()
    i = 0
    j = 0
    res_all = []

    with torch.no_grad():
        for building_id in test_loaders:
            print(datetime.now(), i+1, '/', len(test_loaders), building_id, "Dataset: ", dataset, flush=True)
            # print(datetime.now(), i, '/', len(test_loaders), building_id, "Dataset: ", save_filename, flush=True)
        
            test_loader = test_loaders[building_id]
            trues, preds, histories, losses = [], [], [], []
            
            if (i % 200 == 0) and (i != len(test_loaders) - 1):
                trues, preds, histories, losses = [], [], [], []
                res_all = []
                j += 1
            try:
                for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):

                    timeseries = timeseries.float().to(device)
                    forecast = forecast.float().to(device)
                    input_mask = input_mask.to(device)

                    with torch.amp.autocast(device_type="cuda"):
                        output = model(x_enc = timeseries, input_mask = input_mask)

                    loss = criterion(output.forecast, forecast)
                    losses.append(loss.item())

                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(output.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())
            except Exception as e:
                print(f"Error processing building {building_id}: {e}")
                i += 1
                continue
                # j += 1
                # if j>i:
                #     break
            trues_n = np.concatenate(trues, axis=0)
            preds_n = np.concatenate(preds, axis=0)
            metrics = get_forecasting_metrics(y=trues_n, y_hat=preds_n, reduction='mean')
            # M = len(trues) // 24
            # print(M)
            # y_bar = np.mean(trues)
            # NRMSE = (1/ y_bar) * np.sqrt((1 / (24 * M)) * np.mean((trues - preds) ** 2))
            nrmse = metrics.rmse / np.nanmean(trues_n)
            print(building_id, nrmse)
            res_df = pd.DataFrame({'building': [building_id], 'mae': [metrics.mae], 'mse': [metrics.mse], 'rmse': [metrics.rmse], 'nrmse': [nrmse], 'smape': [metrics.smape]})
            res_all.append(res_df)

            i += 1
            if i % 5 == 0:
                print(datetime.now(), 'Saving...')
                res_all_df = pd.concat(res_all).round(6)
                res_all_df.to_csv(f'../Results/moment/{dtype}/results/{dataset}/metrics_{j}.csv', index=False)

        res_all_df = pd.concat(res_all, ignore_index=True)
        res_all_df.to_csv(f'../Results/moment/{dtype}/results/{dataset}/metrics_{j}.csv', index=False)
        print("Zero-shot completed for ", dataset, ".")     

        return res_all_df

            # print(f"NRMSE for building {building_id}: {nrmse}, NRMSE: {NRMSE}")
            # if j>i:
            #     break


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = "zero-shot using moment")

    parser.add_argument("--dtype", type = str, help = "Distribution")

    args = parser.parse_args()
    dtype = args.dtype
    

    # dir_list = os.listdir(f'/home/samy/Hourly/Datasets/test_new/{dtype}/')
    # dir_list = ['SAVE', 'iFlex', 'SGSC', 'SKC', 'ENERTALK', 'DTH', 'NEEA', 'NESEMP', 'DGS', 'IRH', 'PES', 'LEC', 'HSG', 'UNICON']
    dir_list = ['nrel-com', 'nrel-res']

    for c, dataset in enumerate(dir_list):
        print(f'Model: Moment. Distribution: {dtype}. Dataset {dataset} is started for forecasting. {c+1}/{len(dir_list)}')

        files_list = glob.glob(f'/home/samy/Hourly/Datasets/test_new/{dtype}/{dataset}/*.parquet')
        print(files_list)
        # fg = ['1', '3', '10', '25']
    
        os.makedirs(f'../Results/moment/{dtype}/results/{dataset}/', exist_ok = True)

        # for filename in files_list:
        # print(datetime.now(), files_list)
        results = forecasting(files_list, dataset)
            # if results is not None:
            #     results.to_csv(f'../forecasts/{dataset}/{os.path.basename(filename)}', index=False)
        print('')
        print(f'{c+1}/{len(dir_list)} done.')

    torch.cuda.empty_cache()

