import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lag-llama"))

import glob
from collections import defaultdict
from datetime import datetime
from itertools import islice
import argparse

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.torch.distributions.studentT import StudentTOutput

torch.serialization.add_safe_globals([StudentTOutput])

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

from lag_llama.gluon.estimator import LagLlamaEstimator

import warnings  
warnings.filterwarnings('ignore') 


'''The forecast will be of shape (num_samples, prediction_length), where num_samples is the number of samples sampled
   from the predicted probability distribution for each timestep.'''

def get_lag_llama_predictions(dataset, prediction_length, device, context_length, use_rope_scaling=False, num_samples=100):

    ckpt = torch.load("./lag-llama/lag-llama.ckpt", map_location=device, weights_only=False) # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path="./lag-llama/lag-llama.ckpt",
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


# Data pipelining
def get_batched_data_fn(sub_df,
    batch_size: int = 128, 
    context_len: int = 168, 
    horizon_len: int = 24):
    
    examples = defaultdict(list)
    num_examples = 0
    for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
      num_examples += 1
      examples["inputs"].append(sub_df["y"][start:(context_end := start + context_len)].tolist())
      examples["outputs"].append(sub_df["y"][context_end:(context_end + horizon_len)].tolist())
      examples['inputs_ts'].append(sub_df.index[start:(context_end := start + context_len)])
      examples["outputs_ts"].append(sub_df.index[context_end:(context_end + horizon_len)])

    return examples
  
def forecast_building(df):

    # Set numerical columns as float32
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')
    
    # Create the Pandas
    # dataset = PandasDataset.from_long_dataframe(df.head(30*192), target="target", item_id="item_id")
    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
    
    backtest_dataset = dataset
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 100 # number of samples sampled from the probability distribution for each timestep
    device = torch.device("cuda") # You can switch this to CPU or other GPUs if you'd like, depending on your environment    
    
   # ckpt = torch.load("./lag-llama/lag-llama.ckpt", map_location=device) # Uses GPU since in this Colab we use a GPU.

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

    
def process_building(df): 
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

    res, agg_metrics, ts_metrics = forecast_building(windows_all_df)
    return res, agg_metrics, ts_metrics


# Benchmark
# batch_size = 32
# context_len = 168
# horizon_len = 24

def process_file(filename, dataset):
    df = pd.read_parquet(filename)
    #df = df.set_index(['Timestamp'])
    save_filename = os.path.basename(filename).split(".")[0]

   # if df.shape[1] < 2:
    #    return None
        
    print(datetime.now(), df.shape, flush=True)

    res_all = []
    agg_metrics_all = []
    ts_metrics_all = []
    
    i = 0
    for building_name in df.columns:
        print(datetime.now(), i, '/', len(df.columns), building_name, flush=True)
        df1 = df[[building_name]]#.head(24*200)

        res, agg_metrics, ts_metrics = process_building(df1)
        res['building'] = building_name
        res['filename'] = filename
        res_all.append(res)
        
        ts_metrics.insert(0, 'building', building_name)
        ts_metrics.insert(0, 'filename', filename)
        ts_metrics = ts_metrics.sort_values(['forecast_start'])
        ts_metrics_all.append(ts_metrics)
        
        agg_metrics_df = pd.DataFrame([agg_metrics])
        agg_metrics_df.insert(0, 'building', building_name)
        agg_metrics_df.insert(0, 'filename', filename)
        agg_metrics_all.append(agg_metrics_df)

        i += 1
        if i % 5 == 0:
            print(datetime.now(), 'Saving...')
            res_all_df = pd.concat(res_all).round(6)
            res_all_df = res_all_df.reset_index()
            res_all_df = res_all_df.rename(columns={res_all_df.columns[0]: "timestamp" })
            res_all_df.to_csv(f'/home/user/energygpt/Results/lagllama/forecasts/{dataset}/{save_filename}', index=False)            

            ts_metrics_all_df = pd.concat(ts_metrics_all).round(6)
            ts_metrics_all_df.to_csv(f'/home/user/energygpt/Results/lagllama/results/{dataset}/ts_metrics_{save_filename}', index=False)            

            agg_metrics_all_df = pd.concat(agg_metrics_all).round(6)            
            agg_metrics_all_df.to_csv(f'/home/user/energygpt/Results/lagllama/results/{dataset}/agg_metrics_{save_filename}', index=False)            
    
    
    res_all_df = pd.concat(res_all).round(6)
    res_all_df = res_all_df.reset_index()
    res_all_df = res_all_df.rename(columns={res_all_df.columns[0]: "timestamp" })
    res_all_df.to_csv(f'/home/user/energygpt/Results/lagllama/forecasts/{dataset}/{save_filename}', index=False)            

    ts_metrics_all_df = pd.concat(ts_metrics_all).round(6)    
    ts_metrics_all_df.to_csv(f'/home/user/energygpt/Results/lagllama/results/{dataset}/ts_metrics_{save_filename}', index=False)            

    agg_metrics_all_df = pd.concat(agg_metrics_all).round(6)   
    agg_metrics_all_df.to_csv(f'/home/user/energygpt/Results/lagllama/results/{dataset}/agg_metrics_{save_filename}', index=False)                

    return res_all_df, ts_metrics_all_df, agg_metrics_all_df


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = "zero-shot using lagllama")

    parser.add_argument("--dataset", type = str, help = "Dataset name")

    args = parser.parse_args()
    dataset = args.dataset

    files_list = glob.glob(f'/media/user/DATA/Dataset_V0.0/Energy-Load-Profiles/Hourly/Residential/{dataset}/*.parquet')

    
    os.makedirs(f'/home/user/energygpt/Results/lagllama/forecasts/{dataset}/', exist_ok = True)
    os.makedirs(f'/home/user/energygpt/Results/lagllama/results/{dataset}/', exist_ok = True)

    for filename in files_list:
        print(datetime.now(), filename)
        results = process_file(filename, dataset)
        # if results is not None:
        #     results.to_csv(f'../forecasts/{dataset}/{os.path.basename(filename)}', index=False)
        print('')

    torch.cuda.empty_cache()
   