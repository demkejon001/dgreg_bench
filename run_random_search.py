import argparse
import concurrent.futures

from random_search import train


def get_args():
    parser = argparse.ArgumentParser(description='Domain Generalization for Regression')
    parser.add_argument('-data', '--dataset_type', type=str, default="sin")
    parser.add_argument('-alg', '--algorithm_type', type=str, default="ERM")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[-1])
    parser.add_argument('--log_interval', type=int, default=0, help="Will log training performance at every interval. Default 0 means dataset-dependent.")
    parser.add_argument('--early_stop_threshold', type=int, default=0, help="Number of times val loss can concurrently be lower than the best val loss before stopping early. Default 0 means no early stopping.")
    parser.add_argument('--early_stop_start_step', type=int, default=0, help="Step when early stopping algorithm activates")
    return parser.parse_args()

def run_job(algorithm_type, dataset_type, hparam_seed, log_interval, early_stop_start_step, early_stop_threshold):
    train(algorithm_type, dataset_type, seed=0, hparams_seed=hparam_seed,
          max_steps=0, log_interval=log_interval, early_stop_threshold=early_stop_threshold,
          early_stop_start_step=early_stop_start_step,)
    return hparam_seed


def get_job_args(algorithm_type, dataset_type, log_interval, early_stop_start_step, early_stop_threshold, n_hparam_seeds=60):
    job_args = []
    for hparam_seed in range(n_hparam_seeds):
        job_args.append(( algorithm_type, dataset_type, hparam_seed, log_interval, early_stop_start_step, early_stop_threshold, ))
    return job_args


def main():
    args = get_args()
    n_hparam_seeds = 60
    job_args = get_job_args(args.algorithm_type, args.dataset_type, args.log_interval, args.early_stop_start_step, args.early_stop_threshold, n_hparam_seeds=n_hparam_seeds)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_job, *j_arg) for j_arg in job_args]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__=="__main__":
    main()