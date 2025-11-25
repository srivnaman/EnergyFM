import os
import glob
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness

from torch.utils.data import Dataset, DataLoader

control_randomness(seed=13) 



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
        data_split: str = "train",
        data: str = None,
        divider: int = None
    ):
        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed
        self.data = data
        self.data_split = data_split
        self.divider = divider
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
        
        if self.data_split == "train":
            self.np_data = self.np_data[:self.divider]
        elif self.data_split == "test":
            self.np_data = self.np_data[self.divider:]


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
    train_loaders = {}
    test_loaders = {}
    
    for building_id in building_ids:
        data = df[building_id]
        mid = len(data) // 2
        train_dataset = TSDataset(
            random_seed=13, 
            forecast_horizon=24, 
            data = data,
            data_split="train",
            divider=mid)
        test_dataset = TSDataset(
            random_seed=13, 
            forecast_horizon=24, 
            data = data,
            data_split="test",
            divider=mid
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        train_loaders[building_id] = train_loader
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders[building_id] = test_loader
    return train_loaders, test_loaders


def forecasting(building_id, test_loader):

    # test_loaders = create_building_loaders(files)
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
         
        trues_n = np.concatenate(trues, axis=0)
        preds_n = np.concatenate(preds, axis=0)
        metrics = get_forecasting_metrics(y=trues_n, y_hat=preds_n, reduction='mean')

        nrmse = metrics.rmse / np.nanmean(trues_n)
        print("Pretrained model", building_id, nrmse)
        res_df = pd.DataFrame({'building': [building_id], 'mae': [metrics.mae], 'mse': [metrics.mse], 'rmse': [metrics.rmse], 'nrmse': [nrmse], 'smape': [metrics.smape]})
        res_all.append(res_df)


    res_all_df = pd.concat(res_all, ignore_index=True).round(6)
    # res_all_df.to_csv(f'../Results/moment/{dtype}/results/{dataset}/metrics_{j}.csv', index=False)
    print("Testing completed for ", building_id, ".")     

    return res_all_df



def finetune_forecasting(files, dataset):

    train_loaders, test_loaders = create_building_loaders(files)
    # save_filename = os.path.basename(file).split(".")[0]

    trues, preds, histories, losses = [], [], [], []
    
    i = 0
    j = 0
    res_all = []
    res_finetune_all = []

    for building_id in train_loaders:
        print(datetime.now(), i+1, '/', len(train_loaders), building_id, "Dataset: ", dataset, flush=True)

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)

        criterion = torch.nn.MSELoss()
        criterion = criterion.to(device)

        scaler = torch.amp.GradScaler()

        curr_epoch = 0
        max_epoch = 1

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        max_lr = 1e-4
        max_norm = 5.0 

        try:
            train_loader = train_loaders[building_id]
            total_steps = len(train_loader)*max_epoch

            scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)
            

            while curr_epoch < max_epoch:
                model.train()
                losses = []
                for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):

                    timeseries = timeseries.float().to(device)
                    input_mask = input_mask.to(device)
                    forecast = forecast.float().to(device)

                    with torch.amp.autocast(device_type="cuda"):
                        output = model(x_enc=timeseries, input_mask=input_mask)

                    loss = criterion(output.forecast, forecast)

                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none = True)

                    losses.append(loss.item())

                losses = np.array(losses)
                average_loss = np.nanmean(losses)

                print(f'Epoch {curr_epoch}: Train loss: {average_loss:.3f}')

                scheduler.step()
                curr_epoch += 1
        

            model.eval()
            with torch.no_grad():
                # for building_id in test_loaders:
                # print(datetime.now(), i, '/', len(test_loaders), building_id, "Dataset: ", save_filename, flush=True)
            
                test_loader = test_loaders[building_id]
                trues, preds, histories, losses = [], [], [], []
                
                if (i % 200 == 0) and (i != len(test_loaders) - 1):
                    # trues, preds, histories, losses = [], [], [], []
                    res_all = []
                    res_finetune_all = []
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
        except Exception as e:
            print(f"Error processing building {building_id}: {e}")
            i += 1
            continue
                # j += 1
                # if j>i:
                #     break
        try:
            trues_n = np.concatenate(trues, axis=0)
            preds_n = np.concatenate(preds, axis=0)
        except Exception as e:
            print(f"Error processing building {building_id}: {e}")
            i += 1
            continue
        metrics = get_forecasting_metrics(y=trues_n, y_hat=preds_n, reduction='mean')

        nrmse = metrics.rmse / np.nanmean(trues_n)
        print("Finetuned model", building_id, nrmse)
        res_fine_df = pd.DataFrame({'building': [building_id], 'mae': [metrics.mae], 'mse': [metrics.mse], 'rmse': [metrics.rmse], 'nrmse': [nrmse], 'smape': [metrics.smape]})
        res_df = forecasting(building_id, test_loader)
        res_all.append(res_df)
        res_finetune_all.append(res_fine_df)

        i += 1
        if i % 5 == 0:
            print(datetime.now(), 'Saving...')
            res_all_df = pd.concat(res_all).round(6)
            res_all_df.to_csv(f'../Results/moment-finetune/{dtype}/results/{dataset}/test_metrics_{j}.csv', index=False)
            res_finetune_all_df = pd.concat(res_finetune_all).round(6)
            res_finetune_all_df.to_csv(f'../Results/moment-finetune/{dtype}/results/{dataset}/fine_metrics_{j}.csv', index=False)

    res_all_df = pd.concat(res_all, ignore_index=True)
    res_all_df.to_csv(f'../Results/moment-finetune/{dtype}/results/{dataset}/test_metrics_{j}.csv', index=False)
    res_finetune_all_df = pd.concat(res_finetune_all).round(6)
    res_finetune_all_df.to_csv(f'../Results/moment-finetune/{dtype}/results/{dataset}/fine_metrics_{j}.csv', index=False)
    print("Finetuning completed for ", dataset, ".")     

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
    dir_list = ['BDG-2', 'DGS', 'DTH', 'ECCC', 'ENERTALK', 'HSG', 
                'HUE', 'iFlex', 'IRH', 'LEC', 'NEEA', 'NESEMP', 
                'PES', 'Plegma', 'RSL', 'SAVE', 'SGSC', 'SKC', 'UNICON']
    # dir_list = ['IPC-Commercial']

    for c, dataset in enumerate(dir_list):
        print(f'Model: Moment. Distribution: {dtype}. Dataset {dataset} is started for forecasting. {c+1}/{len(dir_list)}')

        files_list = glob.glob(f'/home/samy/Hourly/Datasets/test_new/{dtype}/{dataset}/*.parquet')
        print(files_list)
        # fg = ['1', '3', '10', '25']
    
        os.makedirs(f'../Results/moment-finetune/{dtype}/results/{dataset}/', exist_ok = True)

        # for filename in files_list:
        # print(datetime.now(), files_list)
        results = finetune_forecasting(files_list, dataset)
            # if results is not None:
            #     results.to_csv(f'../forecasts/{dataset}/{os.path.basename(filename)}', index=False)
        print('')
        print(f'{c+1}/{len(dir_list)} done.')

    torch.cuda.empty_cache()

