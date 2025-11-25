import os
import yaml
import logging
import argparse
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from .models import define_all_classifiers, select_classifiers
from .metrics import compute_metrics

RANDOM_SEED = 42


def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )


def load_data(appliance, data_file, label_file, data_limit):
    df_data = pd.read_csv(data_file)
    if "Timestamp" in df_data.columns:
        df_data = df_data.drop("Timestamp", axis=1)
    df_data = df_data.head(data_limit)
    X = df_data.T.values

    df_labels = pd.read_csv(label_file)
    if appliance not in df_labels.columns:
        logging.error(f"appliance '{appliance}' not found in label file columns: {list(df_labels.columns)}")
        print(f"Error: appliance '{appliance}' not found in label file. See log for details.")
        exit(1)
    y = df_labels[appliance].values
    return X, y


def fit_and_evaluate(appliance, classifiers, X, y, test_size, output_path):    
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED)
    logging.info(f"appliance: {appliance} - training set size: {X_train.shape}, test set size: {X_test.shape}")
    
    logging.info(f"appliance: {appliance} - no. of models: {len(classifiers.keys())} names: {list(classifiers.keys())}")

    all_results = []
    
    tqdm.write(f"appliance: {appliance} - evaluating {len(classifiers.keys())} models. names: {list(classifiers.keys())}")           
    with tqdm(total=len(classifiers.keys()), mininterval=0, miniters=1, desc="Models") as pbar:        
        for clf_name, clf in classifiers.items():
            try:
                logging.info("") 
                logging.info(f"appliance: {appliance} - model: {clf_name} - training")
                clf.fit(X_train, y_train)
                logging.info(f"appliance: {appliance} - model: {clf_name} - predicting")
                y_pred = clf.predict(X_test)

                logging.info(f"appliance: {appliance} - model: {clf_name} - computing metrics")
                metrics = compute_metrics(y_test, y_pred, clf_name)
                logging.info(f"appliance: {appliance} - model: {clf_name} - F1 Score: {metrics.get('F1Score', 'N/A')}")
                logging.info(f"appliance: {appliance} - model: {clf_name} - {metrics}")

                results_df = pd.DataFrame([metrics])
                results_df.insert(0, "Model", clf_name)
                results_df.insert(0, "Appliance", appliance)            
                all_results.append(results_df)                
                pbar.update(1)
                pbar.set_postfix({"Model": clf_name, "F1Score": metrics.get("F1Score", "N/A")})
            except Exception as e:
                logging.error(f"appliance: {appliance} - model: {clf_name} - failed! error: {e}")
                continue
            

    if all_results:
        os.makedirs(output_path, exist_ok=True)
        result_df = pd.concat(all_results)
        result_path = os.path.join(output_path, f"{appliance}.csv")
        result_df.round(6).to_csv(result_path, index=False)
        logging.info("")        
        logging.info(f"appliance: {appliance} - saved results to {result_path}")        
        logging.info("\n" + result_df.to_string(index=False))
        return result_df


def run_experiment(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment_name"]
    data_file = config["data_file"]
    label_file = config["label_file"]
    data_limit = config.get("data_limit", 672)
    test_size = config.get("test_size", 0.3)
    results_dir = config["results_dir"]
    appliance_list = config["appliance_list"]
    model_names = config["models"]

    output_path = os.path.join(results_dir, experiment_name)
    log_path = os.path.join(results_dir, f"{experiment_name}_log.txt")

    setup_logging(log_path)
    
    classifiers = select_classifiers(model_names)
    
    logging.info(f"experiment: {experiment_name} - starting...")
    logging.info("-" * 60 + "\n")

    all_appliance_results = []
    tqdm.write(f"experiment: {experiment_name} - {len(appliance_list)} appliances. names: {appliance_list}")           
    with tqdm(total=len(appliance_list), mininterval=0, miniters=1, desc="Appliances") as pbar:
        for appliance in appliance_list:
            logging.info(f"appliance: {appliance}")    
            logging.info(f"appliance: {appliance} - loading data")
            X, y = load_data(appliance, data_file, label_file, data_limit)
            
            results_appliance = fit_and_evaluate(appliance, classifiers, X, y, test_size, output_path)
            all_appliance_results.append(results_appliance)
            
            logging.info(f"appliance: {appliance} - done")
            logging.info("-" * 60 + "\n")
            pbar.update(1)
            pbar.set_postfix({"appliance": appliance})

    
    if all_appliance_results:
        all_result_df = pd.concat(all_appliance_results)        
        all_results_path = os.path.join(results_dir, f"{experiment_name}_results.csv")
        all_result_df.round(6).to_csv(all_results_path, index=False)        
        logging.info(f"experiment: {experiment_name} - saved combined results to {all_results_path}")        
        logging.info("\n" + all_result_df.to_string(index=False))        
    
    logging.info(f"experiment: {experiment_name} - completed!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NILS classifiers using a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    run_experiment(args.config)
