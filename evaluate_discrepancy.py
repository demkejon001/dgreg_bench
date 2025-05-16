import pandas as pd 
import argparse

from pathlib import Path
from benchmark import BENCHMARK_TYPES
from evaluate_benchmark import get_algorithm_iqm


def get_results_df(run_type, datasets, benchmark_type, pretty_names=True):
    dataset_dfs = []
    for dataset in datasets:
        data_dir = f"data/{run_type}/{benchmark_type}/{dataset}"
        data_dir_path = Path(data_dir)

        dfs = []
        csv_files = data_dir_path.rglob("*.csv")
        for csv_file in csv_files:
            csv_file = str(csv_file)
            _, _, _, _, alg, seed_name, _ = csv_file.split('/')
            trial_seed = seed_name.split("_")[0][1:]
            df = pd.read_csv(csv_file, index_col=0)

            if pretty_names:
                if "_" in alg:
                    alg = alg.replace("_", "-")

            df["alg"] = alg
            df["alg_seed"] = f"{alg}_{trial_seed}"
            df = df.set_index("alg_seed")
            dfs.append(df)

        df = pd.concat(dfs)

        df = df.drop([f"final_tloss"], axis=1)

        if pretty_names:
            if "_" in dataset:
                d1, d2 = dataset.split("_")
                dataset = f"{d1} ({d2})"

        rename_col_map = {"early_stop_tloss": dataset}
        df = df.rename(columns=rename_col_map)

        dataset_dfs.append(df)

    df = pd.concat([df.drop("alg", axis=1) for df in dataset_dfs] + [dataset_dfs[0].alg], axis=1)

    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset_type', type=str, nargs="+", default=[])
    parser.add_argument('--benchmark_type', type=str, choices=BENCHMARK_TYPES, required=True)

    return parser.parse_args()


def main():
    args = get_args()

    datasets = args.dataset_type
    for dataset in datasets:

        oracle_df = get_results_df("oracle", [dataset], args.benchmark_type)
        oracle_df = get_algorithm_iqm(oracle_df)
        
        benchmark_df = get_results_df("benchmark", [dataset], args.benchmark_type)
        benchmark_df = get_algorithm_iqm(benchmark_df)
        generalization_ratio = benchmark_df.loc["ERM"] / oracle_df.loc["ERM"]
        print(f"{dataset} & {generalization_ratio.item():.2f} \\\\")
        print(f"\\hline")


if __name__=="__main__":
    main()