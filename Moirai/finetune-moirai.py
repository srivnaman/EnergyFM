import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/uni2ts')

import os
import torch
import matplotlib.pyplot as plt
import glob
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from tqdm.autonotebook import tqdm
import matplotlib.dates as mdates
from itertools import islice
from collections import defaultdict
import gluonts
from datetime import datetime
import lightning as L

from uni2ts.eval_util.plot import plot_single
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule, MoiraiFinetune
from uni2ts.eval_util.evaluation import evaluate_model
from uni2ts.common.env import env
from pathlib import Path
from uni2ts.data.builder.simple import SimpleEvalDatasetBuilder, generate_eval_builders, SimpleDatasetBuilder
from uni2ts.loss.packed.point import PackedMSELoss, PackedNRMSELoss

import uni2ts
from torch.utils.data import Dataset, Sampler, default_collate, default_convert, DistributedSampler
import itertools
from functools import partial
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import numpy as np
from jaxtyping import Bool, Int
from torch.utils.data import DataLoader as TorchDataLoader
# from torch.utils.data import Dataset, Sampler, default_collate, default_convert
from uni2ts.data.loader import DataLoader, PackCollate
from torch.utils._pytree import tree_map

from uni2ts.common.typing import BatchedSample, Sample
import warnings
import argparse

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

train_batch_size=32
train_batch_size_factor=1.0
train_cycle=True
train_num_batches_per_epoch=50
train_shuffle=False
train_num_workers=1
train_pin_memory=True
train_drop_last=False
train_fill_last=False
train_worker_init_fn=None
train_prefetch_factor=2
train_persistent_workers=True

val_batch_size=32
val_batch_size_factor=2.0
val_cycle=False
val_num_batches_per_epoch=None
val_shuffle=False
val_num_workers=1
val_pin_memory=False
val_drop_last=False
val_fill_last=True
val_worker_init_fn=None
val_prefetch_factor=2
val_persistent_workers=True



# Data pipelining
def get_batched_data_fn(sub_df,
    batch_size: int = 128, 
    context_len: int = 168, 
    horizon_len: int = 24):
    
    examples = defaultdict(list)
    num_examples = 0
    for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
      num_examples += 1
      #examples["country"].append(country)
      examples["inputs"].append(sub_df["y"][start:(context_end := start + context_len)].tolist())
      #examples["gen_forecast"].append(sub_df["gen_forecast"][start:context_end + horizon_len].tolist())
      #examples["week_day"].append(sub_df["week_day"][start:context_end + horizon_len].tolist())
      examples["outputs"].append(sub_df["y"][context_end:(context_end + horizon_len)].tolist())
      examples['inputs_ts'].append(sub_df.index[start:(context_end := start + context_len)])
      examples["outputs_ts"].append(sub_df.index[context_end:(context_end + horizon_len)])

    return examples

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
        model_finetune
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.model = model_finetune

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        model,
        num_batches_per_epoch: Optional[int] = None,
        cycle: bool = False,
        batch_size_factor: float = 1.0,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        fill_last: bool = False,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False     
    ) -> DataLoader:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return dataloader_func(
            dataset=dataset,
            cycle=cycle,
            batch_size_factor=batch_size_factor,
            shuffle=shuffle if sampler is None else None,
            collate_fn = PackCollate(max_length = model.module.max_seq_len, seq_fields=model.seq_fields, pad_func_map=model.pad_func_map),
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            num_workers=2,
            pin_memory=pin_memory,
            drop_last=drop_last,
            fill_last=fill_last,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            DataLoader,
            shuffle = train_shuffle,
            world_size = self.trainer.world_size,
            batch_size = train_batch_size,
            num_batches_per_epoch=train_num_batches_per_epoch,
            cycle=train_cycle,
            batch_size_factor=train_batch_size_factor,
            num_workers = train_num_workers,
            pin_memory = train_pin_memory,
            drop_last = train_drop_last,
            fill_last = train_fill_last,
            worker_init_fn = train_worker_init_fn,
            prefetch_factor = train_prefetch_factor,
            persistent_workers = train_persistent_workers,
            model = self.model)

    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        return tree_map(
            partial(
                self.get_dataloader,
                dataloader_func=DataLoader,
                shuffle=val_shuffle,
                world_size=self.trainer.world_size,
                batch_size=val_batch_size,
                num_batches_per_epoch=val_num_batches_per_epoch,
                cycle=val_cycle,
                batch_size_factor=val_batch_size_factor,
                num_workers = val_num_workers,
                pin_memory = val_pin_memory,
                drop_last = val_drop_last,
                fill_last = val_fill_last,
                worker_init_fn = val_worker_init_fn,
                prefetch_factor = val_prefetch_factor,
                persistent_workers = val_persistent_workers,
                model = self.model),
            self.val_dataset,
        )

    @property
    def train_batch_size(self) -> int:
        return train_batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def val_batch_size(self) -> int:
        return val_batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def train_num_batches_per_epoch(self) -> int:
        return (
            100
            * self.trainer.accumulate_grad_batches
        )

def forecast_building(df):

    # Set numerical columns as float32
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')
    
    # Create the Pandas
    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")

    backtest_dataset = dataset
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 100

    model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
    prediction_length=prediction_length,
    context_length=168,
    patch_size='auto',
    target_dim=1,
    feat_dynamic_real_dim=backtest_dataset.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=backtest_dataset.num_past_feat_dynamic_real,
)
    predictor = model.create_predictor(batch_size=32, device='auto')

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=backtest_dataset,
        predictor=predictor,
        num_samples=num_samples
    )

    # forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    # tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    #print(" forecast Done")
    forecasts = list(forecast_it)
    #print("Done")
    tss = list(ts_it)

    evaluator = Evaluator()
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))

    res_all = []
    for ts, fc in zip(tss, forecasts):
        res = ts[ts.index.isin(fc.index)]
        res.columns = ['y_true']
        res.insert(1, 'y_pred', fc.median)        
        res_all.append(res)
        #print(gt.shape)
        #break
    res_all_df = pd.concat(res_all).sort_index()
    return res_all_df, agg_metrics, ts_metrics 

def finetune_and_forecast_building(building_name, df, filename, dataset_name):#df):

    # Set numerical columns as float32
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')
    
    # Create the Pandas
    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")

    backtest_dataset = dataset
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 100

    model_finetune = MoiraiFinetune(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
    min_patches=2,
    min_mask_ratio=0.15,
    max_mask_ratio=0.5,
    max_dim=128,
    num_training_steps=100,
    num_warmup_steps=0,
    module_kwargs={'patch_sizes': [8, 16, 32, 64, 128]},
    lr=5e-4,
    val_metric = [PackedMSELoss(), PackedNRMSELoss()]
    )

    trainer = L.Trainer(accelerator="gpu",
    # accelerator='auto',
    devices='auto',
    strategy='auto',
    num_nodes=1,
    precision=32,
    logger = L.pytorch.loggers.TensorBoardLogger(save_dir = "", name = "logs"),
    callbacks = [L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch'),
                 L.pytorch.callbacks.ModelCheckpoint(dirpath=f'checkpoints/{dataset_name}/{filename}',
                 monitor='train/PackedNLLLoss',
                 save_weights_only=True,
                 mode='min',
                 save_top_k=1,
                 every_n_epochs=1),
                 L.pytorch.callbacks.EarlyStopping(
          monitor='val/PackedNLLLoss',
          min_delta=0.0,
          patience=3,
          mode='min',
          strict=False,
          verbose=True)],
    max_epochs=3,
    enable_progress_bar=True,
    accumulate_grad_batches=1,
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm'
    )

    train = SimpleDatasetBuilder(dataset=building_name, storage_path = f'Dataset/{dataset_name}/{filename}').load_dataset(model_finetune.train_transform_map)
    L.seed_everything(1 + trainer.logger.version, workers=True)
    
    # val_dataset = SimpleEvalDatasetBuilder(dataset='ETTH_1_eval', storage_path = 'Dataset', prediction_length = 24, context_length = 168, offset = None
    #                                        , windows = 179, distance = 24, patch_size = 32).load_dataset(model_finetune.val_transform_map)
    val =  tree_map(
                lambda ds: ds.load_dataset(model_finetune.val_transform_map),
                generate_eval_builders(dataset=building_name, storage_path = f'Dataset/{dataset_name}/{filename}', offset = 168, eval_length = 24,  prediction_lengths = [24], context_lengths = [168], patch_sizes=[32]))
    
    trainer.fit(
        model_finetune,
        datamodule=DataModule(train, val, model_finetune),
        ckpt_path=None,
    )

    trainer.model.module.save_pretrained(f'models/{dataset_name}/{filename}/{building_name}')

    model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f'models/{dataset_name}/{filename}/{building_name}'),
    #module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
    prediction_length=prediction_length,
    context_length=168,
    patch_size='auto',
    target_dim=1,
    feat_dynamic_real_dim=backtest_dataset.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=backtest_dataset.num_past_feat_dynamic_real,
)
    predictor = model.create_predictor(batch_size=32, device='auto')

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=backtest_dataset,
        predictor=predictor,
        num_samples=num_samples
    )

    # forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    # tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    #print(" forecast Done")
    forecasts = list(forecast_it)
    #print("Done")
    tss = list(ts_it)

    evaluator = Evaluator()
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))

    res_all = []
    for ts, fc in zip(tss, forecasts):
        res = ts[ts.index.isin(fc.index)]
        res.columns = ['y_true']
        res.insert(1, 'y_pred', fc.median)        
        res_all.append(res)
        #print(gt.shape)
        #break
    res_all_df = pd.concat(res_all).sort_index()
    return res_all_df, agg_metrics, ts_metrics 

def window(df):
    building_name = df.columns[0]
    df.columns = ['y']
    input_data = get_batched_data_fn(df, batch_size=500)
    # print(input_data)
    
    windows_all = []
    counter = 1
    for inputs_ts, inputs, outputs_ts, outputs in zip(input_data['inputs_ts'], 
                                                      input_data['inputs'], 
                                                      input_data['outputs_ts'], 
                                                      input_data['outputs']):
        if (not any(inputs)) or (not any(outputs)):
            # print(f"Window {counter} contains all zeros")
            counter += 1
            continue
        
        input_df = pd.DataFrame({'timestamp': inputs_ts, 
                                 'target': inputs})
        
        output_df = pd.DataFrame({'timestamp': outputs_ts, 
                                 'target': outputs})
        combined = pd.concat([input_df, output_df], axis=0)
        combined['item_id'] = str(building_name) + '_' + str(counter)
        combined['item_id_no'] = counter
        counter += 1
        windows_all.append(combined)

    if not windows_all:
        return None
    
    windows_all_df = pd.concat(windows_all)
    windows_all_df.timestamp = pd.to_datetime(windows_all_df.timestamp)
    windows_all_df.set_index('timestamp', inplace=True)

    return windows_all_df


def process_building(train, test, building_name, filename, dataset): 

    # train = windows_all_df[windows_all_df.index <= '2018-06-30 23:00:00']
    # #test = windowed_df[windowed_df.index > '2017-06-30 23:00:00']
    # test = windows_all_df[windows_all_df.item_id_no > train.iloc[-1,:].item_id_no]
    # # train['item_id_no'] = train['item_id_no'].astype('object')
    # # test['item_id_no'] = test['item_id_no'].astype('object')
    os.makedirs(f'Input/{dataset}/{filename}/{building_name}/', exist_ok = True)
    train.drop('item_id_no', axis = 1).to_csv(f'Input/{dataset}/{filename}/{building_name}/Train.csv')
    test.drop('item_id_no', axis = 1).to_csv(f'Input/{dataset}/{filename}/{building_name}/Test.csv')

    SimpleDatasetBuilder(dataset=building_name, storage_path = f'Dataset/{dataset}/{filename}').build_dataset(
        file=Path(f'Input/{dataset}/{filename}/{building_name}/Train.csv'),
        dataset_type='long'
        #freq = 'H'
    )

    SimpleEvalDatasetBuilder(
            dataset=building_name, storage_path = f'Dataset/{dataset}/{filename}',
            offset=None,
            windows=None,
            distance=None,
            prediction_length=24,
            context_length=168,
            patch_size=None,
        ).build_dataset(
            file=Path(f'Input/{dataset}/{filename}/{building_name}/Test.csv'), dataset_type='long', #freq='H'
        )

    
    test_res, test_agg_metrics, test_ts_metrics = forecast_building(test)
    ft_res, ft_agg_metrics, ft_ts_metrics = finetune_and_forecast_building(building_name, test, filename, dataset)#windows_all_df)
    
    return test_res, test_agg_metrics, test_ts_metrics, ft_res, ft_agg_metrics, ft_ts_metrics

def process_file(filename, dataset):
    # df_init = pd.read_csv(filename)
    # df_init = df_init.set_index(['time'])
    # df_init.index = pd.to_datetime(df_init.index)   
    # n = os.path.basename(filename)
    # building_name = n.split('_')[-1].replace('.csv', '')
    # df_init['year'] = df_init.index.year
    # training_set = df_init[df_init.year <= 2015]
    # test_set = df_init[df_init.year > 2015]
    # training_set = training_set.drop(columns='year')
    # test_set = test_set.drop(columns='year')
    # df_init = df_init.drop(columns='year')
    # bs = df_init.columns[:-1].tolist()
    
    df = pd.read_parquet(filename)
    save_filename = os.path.basename(filename).split(".")[0]
        
    test_res_all = []
    test_agg_metrics_all = []
    test_ts_metrics_all = []

    finetuned_res_all = []
    finetuned_agg_metrics_all = []
    finetuned_ts_metrics_all = []
    
    # for year in df_init['year'].unique():
        # b = list(filter(lambda x: str(year) in x, bs))
        # df = df_init[df_init['year'] == year][b]
        # df['month'] = df.index.month
        # training_set = df[df.month <= 6]
        # test_set = df[df.month > 6]
        # training_set = training_set.drop(columns='month')
        # test_set = test_set.drop(columns='month')
        # df = df.drop(columns='month')
    
    # if df_init.shape[1] < 2:
    #     return None
            
    print(datetime.now(), df.shape, flush=True)
    i = 0
    j = 0
    for building_name in df.columns:
        print(datetime.now(), i+1, '/', len(df.columns), building_name, "Dataset: ", dataset, flush=True)
        df1 = df[[building_name]]#.head(24*200)
        print(datetime.now(), i+1, '/', len(df.columns), building_name, "Filename: ", save_filename, df1.shape, flush=True)
        df1 = df1.loc[df1.first_valid_index():]
        print(datetime.now(), i+1, '/', len(df.columns), building_name, "Filename: ", save_filename, df1.shape, flush=True)

        if len(df1) <= 192:
            continue
            
        s = len(df1) // 2
        training_set = df1.iloc[:s, :]
        test_set = df1.iloc[s:, :]
        print(datetime.now(), i+1, '/', len(df.columns), building_name, training_set.shape, flush=True)
        print(datetime.now(), i+1, '/', len(df.columns), building_name, test_set.shape, flush=True)
        # df1 = window(df1)

        # th = df1.item_id_no.max()/2
        train_data = window(training_set)
        test_data  = window(test_set)   

        if (train_data is None) or (test_data is None):
            continue
        # print(datetime.now(), '#items', df1.item_id_no.max(), 'split', th)
        
        # train_data = window(training_set[[building_name]])
        # test_data = window(test_set[[building_name]])

        if (i % 200 == 0) and (i != len(df.columns) - 1):
            test_res_all = []
            test_agg_metrics_all = []
            test_ts_metrics_all = []
        
            finetuned_res_all = []
            finetuned_agg_metrics_all = []
            finetuned_ts_metrics_all = []
            j += 1

        test_res, test_agg_metrics, test_ts_metrics, finetuned_res, finetuned_agg_metrics, finetuned_ts_metrics = process_building(train_data, test_data, building_name, save_filename, dataset)
        test_res['building'] = building_name
        test_res['filename'] = filename
        test_res_all.append(test_res)
        
        test_ts_metrics.insert(0, 'building', building_name)
        test_ts_metrics.insert(0, 'filename', filename)
        test_ts_metrics = test_ts_metrics.sort_values(['forecast_start'])
        test_ts_metrics_all.append(test_ts_metrics)
        
        test_agg_metrics_df = pd.DataFrame([test_agg_metrics])
        test_agg_metrics_df.insert(0, 'building', building_name)
        test_agg_metrics_df.insert(0, 'filename', filename)
        test_agg_metrics_all.append(test_agg_metrics_df)

        finetuned_res['building'] = building_name
        finetuned_res['filename'] = filename
        finetuned_res_all.append(finetuned_res)
        
        finetuned_ts_metrics.insert(0, 'building', building_name)
        finetuned_ts_metrics.insert(0, 'filename', filename)
        finetuned_ts_metrics = finetuned_ts_metrics.sort_values(['forecast_start'])
        finetuned_ts_metrics_all.append(finetuned_ts_metrics)
        
        finetuned_agg_metrics_df = pd.DataFrame([finetuned_agg_metrics])
        finetuned_agg_metrics_df.insert(0, 'building', building_name)
        finetuned_agg_metrics_df.insert(0, 'filename', filename)
        finetuned_agg_metrics_all.append(finetuned_agg_metrics_df)

        i += 1
        if i % 5 == 0:
            test_res_all_df = pd.concat(test_res_all).round(6)
            test_res_all_df = test_res_all_df.reset_index()
            test_res_all_df = test_res_all_df.rename(columns={test_res_all_df.columns[0]: "timestamp" })
            test_res_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/forecasts_test/{dataset}/{save_filename}_{j}.csv', index=False)            

            test_ts_metrics_all_df = pd.concat(test_ts_metrics_all).round(6)
            test_ts_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_test/{dataset}/test_ts_metrics_{save_filename}_{j}.csv', index=False)            

            test_agg_metrics_all_df = pd.concat(test_agg_metrics_all).round(6)            
            test_agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_test/{dataset}/test_agg_metrics_{save_filename}_{j}.csv', index=False)            

            finetuned_res_all_df = pd.concat(finetuned_res_all).round(6)
            finetuned_res_all_df = finetuned_res_all_df.reset_index()
            finetuned_res_all_df = finetuned_res_all_df.rename(columns={finetuned_res_all_df.columns[0]: "timestamp" })
            finetuned_res_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/forecasts_finetuned/{dataset}/{save_filename}_{j}', index=False)            

            finetuned_ts_metrics_all_df = pd.concat(finetuned_ts_metrics_all).round(6)
            finetuned_ts_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_finetuned/{dataset}/finetuned_ts_metrics_{save_filename}_{j}.csv', index=False)            

            finetuned_agg_metrics_all_df = pd.concat(finetuned_agg_metrics_all).round(6)            
            finetuned_agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_finetuned/{dataset}/finetuned_agg_metrics_{save_filename}_{j}.csv', index=False)            

    if (not test_res_all) or (not finetuned_res_all):
        return None
    
    test_res_all_df = pd.concat(test_res_all).round(6)
    test_res_all_df = test_res_all_df.reset_index()
    test_res_all_df = test_res_all_df.rename(columns={test_res_all_df.columns[0]: "timestamp" })
    test_res_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/forecasts_test/{dataset}/{save_filename}_{j}.csv', index=False)            

    test_ts_metrics_all_df = pd.concat(test_ts_metrics_all).round(6)
    test_ts_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_test/{dataset}/test_ts_metrics_{save_filename}_{j}.csv', index=False)            

    test_agg_metrics_all_df = pd.concat(test_agg_metrics_all).round(6)            
    test_agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_test/{dataset}/test_agg_metrics_{save_filename}_{j}.csv', index=False)            

    finetuned_res_all_df = pd.concat(finetuned_res_all).round(6)
    finetuned_res_all_df = finetuned_res_all_df.reset_index()
    finetuned_res_all_df = finetuned_res_all_df.rename(columns={finetuned_res_all_df.columns[0]: "timestamp" })
    finetuned_res_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/forecasts_finetuned/{dataset}/{save_filename}_{j}.csv', index=False)            

    finetuned_ts_metrics_all_df = pd.concat(finetuned_ts_metrics_all).round(6)
    finetuned_ts_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_finetuned/{dataset}/finetuned_ts_metrics_{save_filename}_{j}.csv', index=False)            

    finetuned_agg_metrics_all_df = pd.concat(finetuned_agg_metrics_all).round(6)            
    finetuned_agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_finetuned/{dataset}/finetuned_agg_metrics_{save_filename}_{j}.csv', index=False)     
    print("Finetuning completed for ", dataset, ".")  

    return test_res_all_df, test_ts_metrics_all_df, test_agg_metrics_all_df, finetuned_res_all_df, finetuned_ts_metrics_all_df, finetuned_agg_metrics_all_df


if __name__ == '__main__':
    warnings.filterwarnings('ignore') 

    parser = argparse.ArgumentParser(description = "finetune using moirai")

    parser.add_argument("--dtype", type = str, help = "Distribution")

    args = parser.parse_args()
    dtype = args.dtype

    dir_list = os.listdir(f'/media/user/DATA/Dataset_V0.0/test/{dtype}/')

    for c, dataset in enumerate(dir_list):
        print(f'Model: Moirai Finetuning. Distribution: {dtype}. Dataset {dataset} is started for forecasting. {c+1}/{len(dir_list)}')

        files_list = glob.glob(f'/media/user/DATA/Dataset_V0.0/test/{dtype}/{dataset}/*.parquet')
        # fg = ['1', '3', '10', '25']
    
        os.makedirs(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/forecasts_test/{dataset}/', exist_ok = True)
        os.makedirs(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_test/{dataset}/', exist_ok = True)
        os.makedirs(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/forecasts_finetuned/{dataset}/', exist_ok = True)
        os.makedirs(f'/media/user/DATA/Results/test/{dtype}/moirai-finetune/results_finetuned/{dataset}/', exist_ok = True)

        for filename in files_list:
            print(datetime.now(), filename)
            results = process_file(filename, dataset)
            # if results is not None:
            #     results.to_csv(f'../forecasts/{dataset}/{os.path.basename(filename)}', index=False)
            print('')
        print(f'{c+1}/{len(dir_list)} done.')

    torch.cuda.empty_cache()
    
    # for d in d_list:
    #     print("Dataset - ", d)
    #     files_list = glob.glob(f'/home/user/New_Buildings_Datasets/Test/{d}/*.csv')

    #     dataset = f'{d}-moirai'
    #     os.makedirs(f'forecasts_test/{dataset}/', exist_ok = True)
    #     os.makedirs(f'results_test/{dataset}/', exist_ok = True)
    #     os.makedirs(f'forecasts_finetuned/{dataset}/', exist_ok = True)
    #     os.makedirs(f'results_finetuned/{dataset}/', exist_ok = True)
        
    #     for filename in files_list:
    #         print(datetime.now(), filename)
    #         results = process_file(filename)
    #         # if results is not None:
    #         #     results.to_csv(f'../forecasts/{dataset}/{os.path.basename(filename)}', index=False)
    #         print('')      



