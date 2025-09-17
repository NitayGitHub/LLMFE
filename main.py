"""
Perform Feature Engineering
"""
# Imports
import os
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from utils import is_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

# Arguments
parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
args = parser.parse_args()


if __name__ == '__main__':
    # Define the maximum number of iterations
    global_max_sample_num = 20
    splits = 1
    seed = 42
    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()

    problem_name = args.problem_name
    label_encoder = preprocessing.LabelEncoder()
    is_regression = False
    if problem_name in ['forest-fires', 'housing', 'insurance', 'bike', 'wine', 'crab']:
        is_regression = True

    # Load data observations
    file_name = f"./data/{problem_name}.csv"
    df = pd.read_csv(file_name)
    
    target_attr = df.columns[-1]
    is_cat = [is_categorical(df.iloc[:, i]) for i in range(df.shape[1])][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()
    label_list = np.unique(y).tolist()

    X = X.drop(target_attr, axis=1)

    for col in X.columns:
        if X[col].dtype == 'string':
            X[col] = label_encoder.fit_transform(X[col])


    # Handle missing values
    X = X.fillna(0)
    if is_regression == False:
        y = label_encoder.fit_transform(y)
    else:
        y = y
 
    # Load metadata
    meta_data_name = f"./data/{problem_name}-metadata.json"
    meta_data={}
    try:
        with open(meta_data_name, "r") as f:
            filed_meta_data = json.load(f)
    except:
        filed_meta_data = {}
    meta_data = dict(meta_data, **filed_meta_data)
    
    if splits > 1:
        skf = (
            KFold(n_splits=splits, shuffle=True, random_state=42)
            if is_regression
            else StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        )
        split_generator = skf.split(X, y)
    else:
        test_size = 0.2
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            stratify=y if not is_regression else None,
            random_state=42
        )
        split_generator = [(train_idx, test_idx)]

    for i, (train_idx, test_idx) in enumerate(split_generator, start=1):
        from llmfe import config, sampler, evaluator, pipeline
    
        class_config = config.ClassConfig(
            llm_class=sampler.LocalLLM,
            sandbox_class=evaluator.LocalSandbox
        )
        cfg = config.Config(use_api=args.use_api, api_model=args.api_model)
    
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
        data_dict = {
            'inputs': X_train_fold,
            'outputs': y_train_fold,
            'is_cat': is_cat,
            'is_regression': is_regression
        }
        dataset = {'data': data_dict}
        log_path = args.log_path + f"_split_{i}"
    
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=cfg,
            meta_data=meta_data,
            max_sample_nums=global_max_sample_num * max(1, splits),
            class_config=class_config,
            log_dir=log_path,
        )
