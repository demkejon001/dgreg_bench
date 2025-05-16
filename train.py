import argparse
import json
import os
import numpy as np
import random
import wandb

import torch

import hparams as hparams_registry
import algorithms
from algorithms import Algorithm
from reg_datasets import DATASETS


PROJECT_NAME = "dg_reg"


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

    parser.add_argument('--output_dir', type=str, default="", help="Output directory for logging results. If default will be data/[dataset]/[algorithm]/[seed]")
    parser.add_argument('--save_best_model', action='store_true', help='Will save best model at every val_interval')
    parser.add_argument('--save_last_model', action='store_true', help='Will save model at end of training')
    parser.add_argument('--logger_type', type=str, default="none")
    parser.add_argument('--log_interval', type=int, default=0, help="Will log training performance at every interval. Default 0 means dataset-dependent.")
    parser.add_argument('--val_interval', type=int, default=0, help="Will log val performance at every interval. Default 0 means dataset-dependent.")
    parser.add_argument('--test_interval', type=int, default=0, help="Will log test performance at every interval. Default 0 means no test data evaluation unless log_test is set.")
    parser.add_argument('--log_test', action='store_true', help='Log test performance at end of training')

    parser.add_argument('--early_stop_threshold', type=int, default=0, help="Number of times val loss can concurrently be lower than the best val loss before stopping early. Default 0 means no early stopping.")
    parser.add_argument('--early_stop_start_step', type=int, default=0, help="Step when early stopping algorithm activates")
    parser.add_argument('--job_type', type=str, default=None, help="WandB job type parameter")
    parser.add_argument('--load_model', type=str, default="", help="Pass in filename to load in a model")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopper:
    def __init__(self, start_step, threshold, ):
        self.start_step = start_step
        self.threshold = threshold
        self.best_metric = np.inf
        self.cntr = 0

    def stop_early(self, metric, step):
        if step < self.start_step:
            return False
        else:
            if metric < self.best_metric:
                self.best_metric = metric
                self.cntr = 0
            else:
                self.cntr += 1
                if self.cntr > self.threshold:
                    return True
        return False


class Logger:
    def __init__(self, logger_type, name, hyperparams: dict, dir, job_type=None):
        self.logger_type = logger_type
        self.writer = None
        if logger_type == "wandb":
            wandb.init(project=name, config=hyperparams, dir=dir, job_type=job_type)

    def log_results(self, results: dict, step: int):
        if self.logger_type == "wandb":
            wandb.log(results, step)
        elif self.logger_type == "terminal":
            print(f"Step: {step}")
            for metric_name, val in results.items():
                print(f"{metric_name}: {val}")

    def log_figure(self, name, figure, step):
        if self.logger_type == "wandb":
            im = wandb.Image(figure)
            wandb.log({name: im}, step)

    def close(self):
        if self.logger_type == "wandb":
            wandb.finish()
        

def get_logger(logger_type, hyperparams: dict, dir, job_type) -> Logger:
    # model_type = hyperparams["model_type"],
    # hidden = hyperparams["hidden"]
    # batch_size = hyperparams["batch_size"]
    # n_hidden_layers = hyperparams["n_hidden_layers"]

    if logger_type == "wandb":
        logger = Logger(logger_type, PROJECT_NAME, hyperparams, dir, job_type)
    # elif logger_type == "tensorboard":
    #     exp_name = f"{dir}/{model_type}_h{hidden}_b{batch_size}"
    #     if n_hidden_layers != 2:
    #         exp_name += f"_l{n_hidden_layers}"
    #     logger = Logger(logger_type, exp_name, hyperparams)
    else:
        logger = Logger(logger_type, "none", hyperparams, dir, job_type)
    return logger



def get_logging_intervals(dataset_type, log_interval, val_interval):
    dataset_info = DATASETS[dataset_type]
    log_interval_ = dataset_info.log_interval
    val_interval_ = dataset_info.val_interval

    if log_interval == 0:
        log_interval = log_interval_
    if val_interval == 0:
        val_interval = val_interval_

    return log_interval, val_interval



@torch.no_grad()
def validation_loop(model: Algorithm, data_loader, loss_fn, log_results: dict, is_val: bool):
    data_type = "val" if is_val else "test"
    model.eval()
    for (x, y) in data_loader:
        y_hat = model.predict(x)
        val_results = loss_fn(y_hat, y)
        for k, v in val_results.items():
            if torch.is_tensor(v):
                log_results[f"{data_type}/{k}"] = v.item()
            else:
                log_results[f"{data_type}/{k}"] = v
    model.train()


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
          hparams="",
          seed=0, 
          hparams_seed=0,
          max_steps=0, 
          logger_type="none", 
          output_dir="",
          log_interval = 0,
          val_interval = 0,
          test_interval = 0,
          log_test = False,
          save_best_model = False,
          save_last_model = False,
          early_stop_threshold = 0,
          early_stop_start_step = 0,
          job_type = None,
          load_model = "",
          model_selection_type = "training_domain", 
          test_envs = [-1], 
          val_envs = [], 
          n_val_envs = 1,
          ):

    if output_dir == "":
        output_dir = f"data/{dataset_type}/{algorithm_type}/s{seed}_hs{hparams_seed}"

    if hparams_seed == 0:
        hparams_ = hparams_registry.default_hparams(algorithm_type, dataset_type)
    else:
        hparams_ = hparams_registry.random_hparams(algorithm_type, dataset_type, hparams_seed)
    if hparams:
        hparams_.update(json.loads(hparams))
    hparams = hparams_
    hparams["model_type"] = algorithm_type
    hparams["dataset"] = dataset_type
    batch_size = hparams["batch_size"]
    if max_steps == 0:
        max_steps = hparams["max_steps"]

    set_seed(seed)

    log_interval, val_interval = get_logging_intervals(dataset_type, log_interval, val_interval)
    if test_interval > 0:
        log_test = True
    else:
        test_interval = max_steps + 1  # Will never log


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = "cpu"

    dataset_info = DATASETS[dataset_type]
    if -1 in test_envs:
        test_envs = dataset_info.default_test_envs
    algorithm_class = algorithms.get_algorithm_class(algorithm_type)
    train_loader, val_loader, test_loader, n_domains = dataset_info.get_dataloaders(batch_size, device, seed, model_selection_type, test_envs, val_envs, n_val_envs, hparams)
    loss_fn = dataset_info.loss_fn

    input_shape = dataset_info.input_shape
    n_outputs = dataset_info.n_outputs

    algorithm = algorithm_class(input_shape, n_outputs, n_domains, hparams)
    if load_model != "":
        algorithm.load_state_dict(torch.load(load_model)["state_dict"], strict=False)

    algorithm = algorithm.to(device)

    # TODO: We should probably store alg, data, load_model, early stop, and other hyperparams into hparams for the logger.
    os.makedirs(output_dir, exist_ok=True)
    hparams["alg"] = algorithm_type
    hparams["dataset"] = dataset_type
    logger = get_logger(logger_type, hparams, output_dir, job_type) 

    early_stop_cnt = 0
    best_val_loss = np.inf
    stop_early = False
    if early_stop_threshold <= 0:
        early_stop_threshold = max_steps
    early_stop_start_step = max_steps // 2

    import time
    s = time.time()
    # Training
    step = 0
    while step < max_steps and not stop_early:
        for (x, y) in train_loader:
            # results = algorithm.update(x.to(device), y.to(device), loss_fn)
            results = algorithm.update(x, y, loss_fn)
            step += 1

            log_results = dict()
            if step % log_interval == 0:
                log_results.update(results)

            if step % val_interval == 0:
                validation_loop(algorithm, val_loader, loss_fn, log_results, is_val=True)
                val_loss = log_results["val/loss"]

                if val_loss < best_val_loss:
                    if save_best_model:
                        save_ckpt(algorithm, step, output_dir, "best.ckpt", val_loss)

                    # if log_test:
                        # validation_loop(algorithm, test_loader, loss_fn, log_results, is_val=False)

                    early_stop_cnt = 0
                    best_val_loss = val_loss
                else:
                    if step > early_stop_start_step:
                        early_stop_cnt += 1
                        if early_stop_cnt > early_stop_threshold:
                            stop_early = True

            if step % test_interval == 0:
                validation_loop(algorithm, test_loader, loss_fn, log_results, is_val=False)

            if log_results:
                logger.log_results(log_results, step)

            if step >= max_steps:
                break

            if stop_early:
                break

    # if log_test:
    #     if step % test_interval != 0:  # i.e. if we haven't already logged these results
    #         log_results = dict()
    #         validation_loop(algorithm, test_loader, loss_fn, log_results, is_val=False)
    #         logger.log_results(log_results, step)
    
    if save_last_model:
        save_ckpt(algorithm, step, output_dir, "last.ckpt")

    logger.close()
    print(time.time() - s)


if __name__ == "__main__":
    args = get_args()
    train(**vars(args))
