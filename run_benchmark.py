import argparse
import concurrent.futures

from benchmark import train, BENCHMARK_TYPES


def get_args():
    parser = argparse.ArgumentParser(description='Domain Generalization for Regression')
    parser.add_argument('-data', '--dataset_type', type=str, default="sin", required=True)
    parser.add_argument('-alg', '--algorithm_type', type=str, default="ERM", required=True)
    parser.add_argument('--hparams_seeds', type=int, nargs="+", default=[], help='Seed for random hparams (0 means "default hparams")', required=True)
    parser.add_argument('--benchmark_type', type=str, choices=BENCHMARK_TYPES, required=True)

    parser.add_argument('--n_runs', type=int, default=15, )

    parser.add_argument('--test_envs', type=int, nargs='+', default=[-1])
    parser.add_argument('--log_interval', type=int, default=250, help="Will log training performance at every interval. Default 0 means dataset-dependent.")
    parser.add_argument('--save_best_model', action="store_true")

    parser.add_argument('--early_stop_threshold', type=int, default=0, help="Number of times val loss can concurrently be lower than the best val loss before stopping early. Default 0 means no early stopping.")
    parser.add_argument('--early_stop_start_step', type=int, default=0, help="Step when early stopping algorithm activates")
    return parser.parse_args()


def run_job(algorithm_type, dataset_type, benchmark_type, hparam_seed, seed, log_interval, early_stop_start_step, early_stop_threshold, save_best_model):
    train(algorithm_type, dataset_type, benchmark_type=benchmark_type, seed=seed, hparams_seed=hparam_seed,
          max_steps=0, log_interval=log_interval, early_stop_threshold=early_stop_threshold,
          early_stop_start_step=early_stop_start_step, save_best_model=save_best_model)
    return hparam_seed


def get_job_args(algorithm_type, dataset_type, benchmark_type, hparams_seeds, log_interval, early_stop_start_step, early_stop_threshold, save_best_model, n_runs):
    job_args = []
    if (n_runs % len(hparams_seeds)) != 0:
        raise ValueError(f"{len(hparams_seeds)=} should divide evenly into {n_runs=}")
    n_runs_per_hparam_seed = n_runs // len(hparams_seeds)

    seed = 1
    for hparam_seed in hparams_seeds:
        for _ in range(n_runs_per_hparam_seed):
            job_args.append((algorithm_type, dataset_type, benchmark_type, hparam_seed, seed, log_interval, early_stop_start_step, early_stop_threshold, save_best_model))
            seed += 1
    return job_args


def main():
    args = get_args()
    job_args = get_job_args(algorithm_type=args.algorithm_type, dataset_type=args.dataset_type, 
                            benchmark_type=args.benchmark_type, hparams_seeds=args.hparams_seeds, 
                            log_interval=args.log_interval, early_stop_start_step=args.early_stop_start_step, 
                            early_stop_threshold=args.early_stop_threshold, 
                            save_best_model=args.save_best_model,
                            n_runs=args.n_runs,)

    max_workers = min(10, args.n_runs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, *j_arg) for j_arg in job_args]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__=="__main__":
    main()