import argparse
import json
import os
import pandas as pd
import numpy as np
import torch

import hparams as hparams_registry
import algorithms
from reg_datasets import DATASETS
from train import set_seed, validation_loop


def get_args():
    parser = argparse.ArgumentParser(description='Domain Generalization for Regression')
    parser.add_argument('-data', '--dataset_type', type=str, default="sin")
    parser.add_argument('-alg', '--algorithm_type', type=str, default="ERM")
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--max_steps', type=int, default=0, help='Number of training steps. Default is dataset-dependent.')

    parser.add_argument('--test_envs', type=int, nargs='+', default=[-1])
    parser.add_argument('--val_envs', type=int, nargs='+', default=[])
    parser.add_argument('--model_selection_type', type=str, default="training_domain")
    parser.add_argument('--n_val_envs', type=int, default=1)

    parser.add_argument('--log_interval', type=int, default=0, help="Will log training performance at every interval. Default 0 means dataset-dependent.")

    parser.add_argument('--early_stop_threshold', type=int, default=0, help="Number of times val loss can concurrently be lower than the best val loss before stopping early. Default 0 means no early stopping.")
    parser.add_argument('--early_stop_start_step', type=int, default=0, help="Step when early stopping algorithm activates")
    parser.add_argument('--load_model', type=str, default="", help="Pass in filename to load in a model")
    return parser.parse_args()


def train(algorithm_type, 
          dataset_type, 
          hparams="",
          seed=0, 
          hparams_seed=0,
          max_steps=0, 
          log_interval = 0,
          early_stop_threshold = 0,
          early_stop_start_step = 0,
          load_model = "",
          model_selection_type = "training_domain", 
          test_envs = [-1], 
          val_envs = [], 
          n_val_envs = 1,
          ):

    output_dir = f"data/random_search/{dataset_type}/{algorithm_type}/s{seed}_hs{hparams_seed}"

    if hparams_seed == 0:
        hparams_ = hparams_registry.default_hparams(algorithm_type, dataset_type)
    else:
        hparams_ = hparams_registry.random_hparams(algorithm_type, dataset_type, hparams_seed)
    if hparams:
        hparams_.update(json.loads(hparams))
    hparams = hparams_
    hparams["alg"] = algorithm_type
    hparams["dataset"] = dataset_type
    batch_size = hparams["batch_size"]
    if max_steps == 0:
        max_steps = hparams["max_steps"]

    set_seed(seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = "cpu"

    dataset_info = DATASETS[dataset_type]
    if -1 in test_envs:
        test_envs = dataset_info.default_test_envs
    train_loader, val_loader, _, n_domains = dataset_info.get_dataloaders(batch_size, device, seed, model_selection_type, test_envs, val_envs, n_val_envs, hparams)
    loss_fn = dataset_info.loss_fn
    if log_interval == 0:
        log_interval = dataset_info.log_interval
    input_shape = dataset_info.input_shape
    n_outputs = dataset_info.n_outputs

    algorithm_class = algorithms.get_algorithm_class(algorithm_type)
    algorithm = algorithm_class(input_shape, n_outputs, n_domains, hparams)
    if load_model != "":
        algorithm.load_state_dict(torch.load(load_model)["state_dict"], strict=False)
    algorithm = algorithm.to(device)

    os.makedirs(output_dir, exist_ok=True)

    early_stop_cnt = 0
    stop_early = False
    if early_stop_threshold <= 0:
        early_stop_threshold = max_steps
    best_val_loss = np.inf

    df_dict = dict()

    # Training
    step = 0
    while step < max_steps and not stop_early:
        for (x, y) in train_loader:
            results = algorithm.update(x, y, loss_fn)
            step += 1

            log_results = dict()
            if step % log_interval == 0:
                log_results.update(results)

                validation_loop(algorithm, val_loader, loss_fn, log_results, is_val=True)
                val_loss = log_results["val/loss"]

                if step > early_stop_start_step:
                    if val_loss < best_val_loss:
                        early_stop_cnt = 0
                        best_val_loss = val_loss
                    else:
                        early_stop_cnt += 1
                        if early_stop_cnt > early_stop_threshold:
                            stop_early = True
                
                for key, value in log_results.items():
                    if key not in df_dict:
                        df_dict[key] = []
                    if torch.is_tensor(value):
                        df_dict[key].append(value.item())
                    else:
                        df_dict[key].append(value)

            if step >= max_steps or stop_early:
                break

    df = pd.DataFrame(df_dict)
    df.to_csv(f"{output_dir}/results.csv")


if __name__ == "__main__":
    args = get_args()
    train(**vars(args))