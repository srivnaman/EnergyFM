import os
import glob
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from tqdm.autonotebook import tqdm

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd
import argparse


# Add Lag-Llama to path
sys.path.append(os.path.join(os.path.dirname(__file__), "lag-llama"))

from lag_llama.gluon.estimator import LagLlamaEstimator

torch.set_float32_matmul_precision("medium")
import warnings  
warnings.filterwarnings('ignore') 


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

def window(df):
    building_name = df.columns[0]
    df.columns = ['y']
    input_data = get_batched_data_fn(df, batch_size=500)
    
    windows_all = []
    counter = 1
    for inputs_ts, inputs, outputs_ts, outputs in zip(input_data['inputs_ts'], 
                                                      input_data['inputs'], 
                                                      input_data['outputs_ts'], 
                                                      input_data['outputs']):
        
        input_df = pd.DataFrame({'timestamp': inputs_ts, 
                                 'target': inputs})
        
        output_df = pd.DataFrame({'timestamp': outputs_ts, 
                                 'target': outputs})
        combined = pd.concat([input_df, output_df], axis=0)
        combined['item_id'] = str(building_name) + '_' + str(counter)
        combined['item_id_no'] = counter
        counter += 1
        windows_all.append(combined)
        
    windows_all_df = pd.concat(windows_all)
    windows_all_df.timestamp = pd.to_datetime(windows_all_df.timestamp)
    windows_all_df.set_index('timestamp', inplace=True)

    return windows_all_df

def get_lag_llama_predictions(dataset, prediction_length, device, context_length, use_rope_scaling=False, num_samples=100):
    ckpt = torch.load("./lag-llama/checkpoints/lag-llama.ckpt", map_location=device) # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path="./lag-llama/checkpoints/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

        batch_size=32,
        num_parallel_samples=10,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


def forecast_building(df):

    # Set numerical columns as float32
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')
    
    # Create the Pandas
    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
    #dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
    
    backtest_dataset = dataset
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 10 # number of samples sampled from the probability distribution for each timestep
    device = torch.device("cuda:0") # You can switch this to CPU or other GPUs if you'd like, depending on your environment    
    

    forecasts, tss = get_lag_llama_predictions(backtest_dataset, prediction_length, device, context_length=168, num_samples=num_samples)

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


def fine_tune_and_forecast_building(train, test, dataset):

    prediction_length = 24
    context_length = 168
    num_samples = 20
    device = "cuda"
    
    ckpt = torch.load(f"./lag-llama/checkpoints_bdg/lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    
    estimator = LagLlamaEstimator(
            ckpt_path=f"./lag-llama/checkpoints_bdg/lag-llama.ckpt",
            prediction_length=prediction_length,
            context_length=context_length,
    
            # distr_output="neg_bin",
            # scaling="mean",
            nonnegative_pred_samples=True,
            aug_prob=0,
            lr=5e-4,
    
            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=estimator_args["time_feat"],
    
            # rope_scaling={
            #     "type": "linear",
            #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            # },
    
            batch_size=64,
            num_parallel_samples=num_samples,
            trainer_kwargs = {"max_epochs": 3,}, # <- lightning trainer arguments
        )    
    
    # Create the Pandas
    dataset_train = PandasDataset.from_long_dataframe(train, target="target", item_id="item_id")    
    predictor = estimator.train(dataset_train, cache_data=True, shuffle_buffer_length=1000)    
    #return predictor

    # Create the Pandas
    dataset_test = PandasDataset.from_long_dataframe(test, target="target", item_id="item_id")    
    forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset_test,
            predictor=predictor,
            num_samples=num_samples
        )

    forecasts = list(forecast_it)
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

def process_building(train, test, dataset): 
    
    # Set numerical columns as float32
    for col in train.columns:
        # Check if column is not of string type
        if train[col].dtype != 'object' and pd.api.types.is_string_dtype(train[col]) == False:
            train[col] = train[col].astype('float32')

    for col in test.columns:
        # Check if column is not of string type
        if test[col].dtype != 'object' and pd.api.types.is_string_dtype(test[col]) == False:
            test[col] = test[col].astype('float32')
    
    
    test_res, test_agg_metrics, test_ts_metrics = forecast_building(test)
    ft_res, ft_agg_metrics, ft_ts_metrics = fine_tune_and_forecast_building(train, test, dataset)
    return test_res, test_agg_metrics, test_ts_metrics, ft_res, ft_agg_metrics, ft_ts_metrics


def process_file(filename, dataset):
    df = pd.read_parquet(filename)
    save_filename = os.path.basename(filename).split(".")[0]

    # if df.shape[1] < 2:
    #     return None
        
    print(datetime.now(), df.shape, flush=True)

    test_res_all = []
    test_agg_metrics_all = []
    test_ts_metrics_all = []

    finetuned_res_all = []
    finetuned_agg_metrics_all = []
    finetuned_ts_metrics_all = []
    
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

        if (i % 200 == 0) and (i != len(df.columns) - 1):
            test_res_all = []
            test_agg_metrics_all = []
            test_ts_metrics_all = []
        
            finetuned_res_all = []
            finetuned_agg_metrics_all = []
            finetuned_ts_metrics_all = []
            j += 1


        test_res, test_agg_metrics, test_ts_metrics, finetuned_res, finetuned_agg_metrics, finetuned_ts_metrics = process_building(train_data, test_data, dataset)
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
            print(datetime.now(), 'Saving...')

            test_res_all_df = pd.concat(test_res_all).round(6)
            test_res_all_df = test_res_all_df.reset_index()
            test_res_all_df = test_res_all_df.rename(columns={test_res_all_df.columns[0]: "timestamp" })
            test_res_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/forecasts_finetuned/{dataset}/test_{save_filename}_{j}.csv', index=False)            

            test_ts_metrics_all_df = pd.concat(test_ts_metrics_all).round(6)
            test_ts_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/test_ts_metrics_{save_filename}_{j}.csv', index=False)            

            test_agg_metrics_all_df = pd.concat(test_agg_metrics_all).round(6)            
            test_agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/test_agg_metrics_{save_filename}_{j}.csv', index=False)            

            finetuned_res_all_df = pd.concat(finetuned_res_all).round(6)
            finetuned_res_all_df = finetuned_res_all_df.reset_index()
            finetuned_res_all_df = finetuned_res_all_df.rename(columns={finetuned_res_all_df.columns[0]: "timestamp" })
            finetuned_res_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/forecasts_finetuned/{dataset}/finetuned_{save_filename}_{j}.csv', index=False)            

            finetuned_ts_metrics_all_df = pd.concat(finetuned_ts_metrics_all).round(6)
            finetuned_ts_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/finetuned_ts_metrics_{save_filename}_{j}.csv', index=False)            

            finetuned_agg_metrics_all_df = pd.concat(finetuned_agg_metrics_all).round(6)            
            finetuned_agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/finetuned_agg_metrics_{save_filename}_{j}.csv', index=False)            
    
    test_res_all_df = pd.concat(test_res_all).round(6)
    test_res_all_df = test_res_all_df.reset_index()
    test_res_all_df = test_res_all_df.rename(columns={test_res_all_df.columns[0]: "timestamp" })
    test_res_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/forecasts_finetuned/{dataset}/test_{save_filename}_{j}.csv', index=False)            

    test_ts_metrics_all_df = pd.concat(test_ts_metrics_all).round(6)
    test_ts_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/test_ts_metrics_{save_filename}_{j}.csv', index=False)            

    test_agg_metrics_all_df = pd.concat(test_agg_metrics_all).round(6)            
    test_agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/test_agg_metrics_{save_filename}_{j}.csv', index=False)            

    finetuned_res_all_df = pd.concat(finetuned_res_all).round(6)
    finetuned_res_all_df = finetuned_res_all_df.reset_index()
    finetuned_res_all_df = finetuned_res_all_df.rename(columns={finetuned_res_all_df.columns[0]: "timestamp" })
    finetuned_res_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/forecasts_finetuned/{dataset}/finetuned_{save_filename}_{j}.csv', index=False)            

    finetuned_ts_metrics_all_df = pd.concat(finetuned_ts_metrics_all).round(6)
    finetuned_ts_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/finetuned_ts_metrics_{save_filename}_{j}.csv', index=False)            

    finetuned_agg_metrics_all_df = pd.concat(finetuned_agg_metrics_all).round(6)            
    finetuned_agg_metrics_all_df.to_csv(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/finetuned_agg_metrics_{save_filename}_{j}.csv', index=False)            

    return test_res_all_df, test_ts_metrics_all_df, test_agg_metrics_all_df, finetuned_res_all_df, finetuned_ts_metrics_all_df, finetuned_agg_metrics_all_df



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finetune using lagllama")
    parser.add_argument("--dtype", type=str, help="Distribution")
    args = parser.parse_args()

    dtype = args.dtype
    # files_list = glob.glob(f'/media/user/DATA/Dataset_V0.0/Energy-Load-Profiles/Hourly/Residential/{dataset}/*.parquet')

    # os.makedirs(f'/home/user/energygpt/Results_nurips/lagllama/forecasts/{dataset}/', exist_ok=True)
    # os.makedirs(f'/home/user/energygpt/Results_nurips/lagllama/results/{dataset}/', exist_ok=True)
    dir_list = os.listdir(f'/media/user/DATA/Dataset_V0.0/test/{dtype}/')

    for c, dataset in enumerate(dir_list):
        print(f'Model: Lag-Llama. Distribution: {dtype}. Dataset {dataset} is started for forecasting. {c+1}/{len(dir_list)}')

        files_list = glob.glob(f'/media/user/DATA/Dataset_V0.0/test/{dtype}/{dataset}/*.parquet')
        # fg = ['1', '3', '10', '25']
    
        os.makedirs(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/forecasts_finetuned/{dataset}/', exist_ok = True)
        os.makedirs(f'/media/user/DATA/Results/test/{dtype}/lag-llama-finetune/results_finetuned/{dataset}/', exist_ok = True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
        for filename in files_list:
            print(datetime.now(), filename)
            results = process_file(filename, dataset)
            print('')
        print(f'{c+1}/{len(dir_list)} done.')
    
        torch.cuda.empty_cache()
    