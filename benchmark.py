import argparse
import json
import os
import numpy as np
import pandas as pd

import torch
import torch.utils.data

import hparams as hparams_registry
import algorithms
from algorithms import Algorithm
from reg_datasets import DATASETS
from train import set_seed, validation_loop


BENCHMARK_TYPES = ["split", "global"]


class EarlyStopTracker:
    def __init__(self, start_step):
        self.start_step = start_step
        self.early_stop_metric = np.inf
        self.tracked_metric = np.inf
        self.cntr = 0

    def update(self, early_stop_metric, tracked_metric, step):
        updated = False
        if step < self.start_step:
            return False

        if early_stop_metric < self.early_stop_metric:
            updated = True
            self.early_stop_metric = early_stop_metric
            self.tracked_metric = tracked_metric
            self.cntr = 0
        else:
            self.cntr += 1
        return updated


def get_args():
    parser = argparse.ArgumentParser(description='Domain Generalization for Regression')
    parser.add_argument('-data', '--dataset_type', type=str, default="sin")
    parser.add_argument('-alg', '--algorithm_type', type=str, default="ERM")
    parser.add_argument('--benchmark_type', type=str, choices=BENCHMARK_TYPES)
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--max_steps', type=int, default=0, help='Number of training steps. Default is dataset-dependent.')

    parser.add_argument('--test_envs', type=int, nargs='+', default=[-1])
    parser.add_argument('--val_envs', type=int, nargs='+', default=[])
    parser.add_argument('--model_selection_type', type=str, default="training_domain")
    parser.add_argument('--n_val_envs', type=int, default=1)

    parser.add_argument('--save_best_model', action='store_true', help='Will save best model at every val_interval')
    parser.add_argument('--save_last_model', action='store_true', help='Will save model at end of training')
    parser.add_argument('--log_interval', type=int, default=0, help="Will log training performance at every interval. Default 0 means dataset-dependent.")

    parser.add_argument('--early_stop_start_step', type=int, default=0, help="Step when early stopping algorithm activates")
    parser.add_argument('--early_stop_threshold', type=int, default=0,)
    parser.add_argument('--load_model', type=str, default="", help="Pass in filename to load in a model")
    return parser.parse_args()


def get_logging_intervals(dataset_type, log_interval, val_interval):
    dataset_info = DATASETS[dataset_type]
    log_interval_ = dataset_info.log_interval
    val_interval_ = dataset_info.val_interval

    if log_interval == 0:
        log_interval = log_interval_
    if val_interval == 0:
        val_interval = val_interval_

    return log_interval, val_interval


def save_ckpt(algorithm: Algorithm, step: int, output_dir: str, name: str, val_loss = None):
    save_ckpt = {
        "step": step,
        "state_dict": algorithm.state_dict(),
    }
    if val_loss is not None:
        save_ckpt["val_loss"] = val_loss
    torch.save(save_ckpt, os.path.join(output_dir, name))
    del save_ckpt


def train(algorithm_type, 
          dataset_type, 
          benchmark_type,
          hparams="",
          seed=0, 
          hparams_seed=0,
          max_steps=0, 
          log_interval = 0,
          save_best_model = False,
          save_last_model = False,
          early_stop_start_step = 0,
          early_stop_threshold = 0,
          load_model = "",
          model_selection_type = "training_domain", 
          test_envs = [-1], 
          val_envs = [], 
          n_val_envs = 1,
          ):

    if benchmark_type not in BENCHMARK_TYPES:
        raise ValueError(f"{benchmark_type=} not found in {BENCHMARK_TYPES=}")

    output_dir = f"data/benchmark/{benchmark_type}/{dataset_type}/{algorithm_type}/s{seed}_hs{hparams_seed}"
    if model_selection_type == "discrepancy":
        output_dir = f"data/oracle/{benchmark_type}/{dataset_type}/{algorithm_type}/s{seed}_hs{hparams_seed}"

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

    dataset_info = DATASETS[dataset_type]
    if -1 in test_envs:
        test_envs = dataset_info.default_test_envs
    train_loader, val_loader, test_loader, n_domains = dataset_info.get_dataloaders(batch_size, device, seed, model_selection_type, test_envs, val_envs, n_val_envs, hparams)
    loss_fn = dataset_info.loss_fn
    if log_interval == 0:
        log_interval = dataset_info.val_interval
    input_shape = dataset_info.input_shape
    n_outputs = dataset_info.n_outputs

    algorithm_class = algorithms.get_algorithm_class(algorithm_type)
    algorithm = algorithm_class(input_shape, n_outputs, n_domains, hparams)
    if load_model != "":
        algorithm.load_state_dict(torch.load(load_model)["state_dict"], strict=False)

    algorithm = algorithm.to(device)

    os.makedirs(output_dir, exist_ok=True)

    early_stop_tracker = EarlyStopTracker(early_stop_start_step)

    early_stop_cnt = 0
    stop_early = False
    if early_stop_threshold <= 0:
        early_stop_threshold = max_steps
    best_val_loss = np.inf

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
                validation_loop(algorithm, test_loader, loss_fn, log_results, is_val=False)
                val_loss = log_results["val/loss"]
                test_loss = log_results["test/loss"]

                if step > early_stop_start_step:
                    if val_loss < best_val_loss:
                        early_stop_cnt = 0
                        best_val_loss = val_loss
                    else:
                        early_stop_cnt += 1
                        if early_stop_cnt > early_stop_threshold:
                            stop_early = True

                val_loss_improved = early_stop_tracker.update(val_loss, test_loss, step)
                if val_loss_improved:
                    if save_best_model:
                        save_ckpt(algorithm, step, output_dir, "best.ckpt", val_loss)

            if step >= max_steps or stop_early:
                break

    df = pd.DataFrame({"final_tloss": [test_loss], "early_stop_tloss": [early_stop_tracker.tracked_metric]})
    df.to_csv(f"{output_dir}/results.csv")
    
    if save_last_model:
        save_ckpt(algorithm, step, output_dir, "last.ckpt")


if __name__ == "__main__":
    args = get_args()
    train(**vars(args))
