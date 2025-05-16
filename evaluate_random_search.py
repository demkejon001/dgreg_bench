import pandas as pd 
import argparse
from pathlib import Path


RUNTIMES = {
    "poverty_income": {
        "ERM": "0:30:00",
        "SD": "0:30:00",
        "VREx": "0:35:00",
        "IB_ERM": "0:40:00",
        "RDM": "1:20:00",
        "IRM": "1:20:00",
        "IB_IRM": "1:20:00",
        "GroupDRO": "0:30:00",
        "EQRM": "4:00:00",
        "CausIRL_CORAL": "0:50:00",
        "CausIRL_MMD": "1:00:00",
        "ANDMask": "1:20:00",
        "SANDMask": "1:40:00",
        "Fish": "2:20:00",
        "CORAL": "1:20:00",
        "MMD": "1:00:00",
        "IGA": "1:50:00",
        "DAEL": "1:50:00",
    },
    "sin": {
        "ERM": "2:00:00",
        "SD": "2:00:00",
        "VREx": "2:30:00",
        "IB_ERM": "2:00:00",
        "RDM": "4:00:00",
        "IRM": "4:00:00",
        "IB_IRM": "4:00:00",
        "GroupDRO": "3:00:00",
        "EQRM": "12:00:00",
        "CausIRL_CORAL": "4:00:00",
        "CausIRL_MMD": "4:00:00",
        "ANDMask": "4:00:00",
        "SANDMask": "12:00:00",
        "Fish": "8:00:00",
        "CORAL": "4:00:00",
        "MMD": "4:00:00",
        "IGA": "6:00:00",
        "DAEL": "6:00:00",
    },
    "nonlin_hard": {
        "ERM": "0:50:00",
        "SD": "0:50:00",
        "VREx": "1:00:00",
        "IB_ERM": "1:10:00",
        "RDM": "2:00:00",
        "IRM": "2:00:00",
        "IB_IRM": "2:00:00",
        "GroupDRO": "0:50:00",
        "EQRM": "8:00:00",
        "CausIRL_CORAL": "1:20:00",
        "CausIRL_MMD": "1:30:00",
        "ANDMask": "2:00:00",
        "SANDMask": "2:40:00",
        "Fish": "4:00:00",
        "CORAL": "2:00:00",
        "MMD": "1:30:00",
        "IGA": "3:00:00",
        "DAEL": "3:00:00",
    },
    "bike_room_day_room_time": {
        "ERM": "0:25:00",
        "SD": "0:25:00",
        "VREx": "0:30:00",
        "IB_ERM": "0:35:00",
        "RDM": "1:00:00",
        "IRM": "1:00:00",
        "IB_IRM": "1:00:00",
        "GroupDRO": "0:25:00",
        "EQRM": "4:00:00",
        "CausIRL_CORAL": "0:40:00",
        "CausIRL_MMD": "0:50:00",
        "ANDMask": "1:00:00",
        "SANDMask": "2:00:00",
        "Fish": "2:00:00",
        "CORAL": "1:00:00",
        "MMD": "0:50:00",
        "IGA": "1:30:00",
        "DAEL": "1:30:00",
    },
    "taxi_accident": {
        "ERM": "0:50:00",
        "SD": "0:50:00",
        "VREx": "1:00:00",
        "IB_ERM": "1:10:00",
        "RDM": "2:00:00",
        "IRM": "2:00:00",
        "IB_IRM": "2:00:00",
        "GroupDRO": "0:50:00",
        "EQRM": "8:00:00",
        "CausIRL_CORAL": "1:20:00",
        "CausIRL_MMD": "1:30:00",
        "ANDMask": "2:00:00",
        "SANDMask": "2:40:00",
        "Fish": "4:00:00",
        "CORAL": "2:00:00",
        "MMD": "1:30:00",
        "IGA": "3:00:00",
        "DAEL": "3:00:00",
    },
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset_type', type=str, default="nonlin")
    parser.add_argument('-k', type=int, default=5, help="Used to compute rolling average of val losses")

    return parser.parse_args()


def get_algs_argmin_seed(df, metric):
    min_loss_df = df.groupby(['seed', 'alg'], as_index=False)[metric].min()
    return min_loss_df.loc[min_loss_df.groupby('alg')[metric].idxmin()]


def get_algs_argmin_seed_split(df, metric, seed_group_size):
    max_seed = df.seed.max()
    min_loss_df_seeds = []
    for seed_start in range(0, max_seed, seed_group_size):
        df_seed = df[(df.seed >= seed_start) & (df.seed < (seed_start+seed_group_size))]
        min_loss_df_seeds.append(get_algs_argmin_seed(df_seed, metric))
    return min_loss_df_seeds


def print_benchmark_script(min_loss_df_seeds: list[pd.DataFrame], dataset_type):
    min_loss_df = min_loss_df_seeds[0]
    algs = min_loss_df.alg.to_numpy()

    alg_seeds = dict()
    for alg in algs:
        alg_seeds[alg] = []
        for min_loss_df in min_loss_df_seeds:
            seed = min_loss_df[min_loss_df.alg == alg].seed.item()
            alg_seeds[alg].append(str(seed))


    if len(min_loss_df_seeds) > 1:
        benchmark_type = "split"
    else:
        benchmark_type = "global"

    
    hseeds = " ".join(['"' + " ".join(alg_seeds[alg]) + '"' for alg in algs])
    algs = " ".join(algs)

    print(f"algs=({algs})")
    print(f"hseeds=({hseeds})")
    print( 'for i in "${!algs[@]}"; do')
    print( '  hseed="${hseeds[$i]}"')
    print( '  alg="${algs[$i]}"')
    print(f'  python run_benchmark.py --alg $alg --data {dataset_type} --hparam_seeds $hseed --benchmark_type {benchmark_type} --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40')
    print(f'done')


def print_sbatch_benchmark_script(min_loss_df_seeds: list[pd.DataFrame], dataset_type):
    min_loss_df = min_loss_df_seeds[0]
    algs = min_loss_df.alg.to_numpy()

    alg_seeds = dict()
    for alg in algs:
        alg_seeds[alg] = []
        for min_loss_df in min_loss_df_seeds:
            seed = min_loss_df[min_loss_df.alg == alg].seed.item()
            alg_seeds[alg].append(str(seed))

    if len(min_loss_df_seeds) > 1:
        benchmark_type = "split"
    else:
        benchmark_type = "global"

    alg_to_runtime = None
    for key in RUNTIMES.keys():
        if dataset_type in key:
            alg_to_runtime = RUNTIMES[key]
            break
    if alg_to_runtime is None:
        raise ValueError(f"Could not recognize {dataset_type} in RUNTIMES keys={list(RUNTIMES.keys())}")
    
    runtime_str = " ".join(['"' + alg_to_runtime[alg] + '"' for alg in algs])
    hseeds = " ".join(['"' + " ".join(alg_seeds[alg]) + '"' for alg in algs])
    algs = " ".join(algs)

    print(f"data={dataset_type}")
    print(f"ALGS=({algs})")
    print(f"RUNTIMES=({runtime_str})")
    print(f"HSEEDS=({hseeds})")
    print( 'for i in ${!ALGS[@]}; do')
    print( '  alg=${ALGS[$i]}')
    print( '  time=${RUNTIMES[$i]}')
    print( '  hseed="${HSEEDS[$i]}"')
    print( '  sbatch --job-name=bench_$alg$data \\')
    print( '    --time=$time \\')
    print( '    --gpus=1 \\')
    print( '    --ntasks=10 \\')
    print( '    --mem-per-cpu="1024M" \\')
    print(f'    --wrap="OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE python run_benchmark.py --alg $alg --data $data --hparam_seeds $hseed --benchmark_type {benchmark_type} --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40"')
    print( '  sleep 1')
    print(f'done')


def main():
    args = get_args()
    data_dir = f"data/random_search/{args.dataset_type}"
    data_dir_path = Path(data_dir)

    dfs = []
    csv_files = data_dir_path.rglob("*.csv")
    columns = ["val/loss", "alg", "seed"]
    for csv_file in csv_files:
        csv_file = str(csv_file)
        alg = csv_file.split('/')[3]
        seed = csv_file.split('/')[4].split('_')[1].split("hs")[-1]
        df = pd.read_csv(csv_file, index_col=0)
        df["alg"] = alg
        df["seed"] = int(seed)
        dfs.append(df[columns])

    df = pd.concat(dfs, ignore_index=True)
        
    k = args.k
    df['rolling_avg'] = df.groupby(['seed', 'alg'])['val/loss'].rolling(window=k, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    print("#!/bin/bash\n")
    min_loss_df_seeds = get_algs_argmin_seed_split(df, "rolling_avg", 20)
    # print_benchmark_script(min_loss_df_seeds, args.dataset_type)
    print_sbatch_benchmark_script(min_loss_df_seeds, args.dataset_type)
    print('')

    min_running_avg_loss_df = get_algs_argmin_seed(df, "rolling_avg")
    # print_benchmark_script([min_running_avg_loss_df], args.dataset_type)
    print_sbatch_benchmark_script([min_running_avg_loss_df], args.dataset_type)


if __name__=="__main__":
    main()
