import os
import argparse
import pickle
from pathlib import Path
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.patches as patches
import seaborn as sns

import scipy
import scipy.stats
from scipy.stats import trim_mean

from rliable.metrics import aggregate_iqm
from rliable.library import get_interval_estimates
from rliable import plot_utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset_type', type=str, nargs="+", default=[])
    return parser.parse_args()


# Modified from rliable. Changed alternative to "less"
def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray):
  num_tasks = scores_x.shape[1]
  task_improvement_probabilities = []
  num_runs_x, num_runs_y = scores_x.shape[0], scores_y.shape[0]
  for task in range(num_tasks):
    if np.array_equal(scores_x[:, task], scores_y[:, task]):
      task_improvement_prob = 0.5
    else:
      task_improvement_prob, _ = scipy.stats.mannwhitneyu(
          scores_x[:, task], scores_y[:, task], alternative='less')
      task_improvement_prob /= (num_runs_x * num_runs_y)
    task_improvement_probabilities.append(task_improvement_prob)
  return np.mean(task_improvement_probabilities)


def get_mann_whitney_u_bci(datasets, benchmark_type):
    df = get_benchmark_df(datasets, benchmark_type)
    datasets = list(df.columns[:-1]) + ["aggregate"]

    alg_dict = dict()
    algs = []
    alg_to_idx = dict()
    for i, (alg, df_alg) in enumerate(df.groupby("alg")):
        alg_dict[alg] = df_alg.drop("alg", axis=1).to_numpy()
        algs.append(alg)
        alg_to_idx[alg] = i
    algs = np.sort(algs)

    alg_pair_dict = dict()
    for alg, scores in alg_dict.items():
        # if alg == "ERM":
            # continue
        alg_pair_dict[f"{alg},ERM"] = (scores, alg_dict["ERM"])

    aggregate_func = lambda x, y: np.array([probability_of_improvement(x[:, i:i+1], y[:, i:i+1]) for i in range(len(datasets) - 1)] + [probability_of_improvement(x, y)])
    # scores, score_cis = get_interval_estimates(alg_pair_dict, aggregate_func, reps=2000)
    scores, score_cis = get_interval_estimates(alg_pair_dict, aggregate_func, reps=50000)

    results_dict = {
            "u": np.full((len(algs), len(datasets)), -1.0),
            "lower_ci": np.full((len(algs), len(datasets)), -1.0),
            "upper_ci": np.full((len(algs), len(datasets)), -1.0),
        }

    for alg_pair in scores:
        alg_a, alg_b = alg_pair.split(",")
        a_idx = alg_to_idx[alg_a]
        for i, dataset in enumerate(datasets):
            results_dict["u"][a_idx, i] = scores[alg_pair][i]
            results_dict["lower_ci"][a_idx, i] = score_cis[alg_pair][0, i]
            results_dict["upper_ci"][a_idx, i] = score_cis[alg_pair][1, i]

    for dict_type, values in results_dict.items():
        results_dict[dict_type] = pd.DataFrame(values, index=algs, columns=datasets)

    with open(f"data/mann_whit_{benchmark_type}.pkl", "wb") as file:
        pickle.dump(results_dict, file)


def get_iqm_bci(datasets, benchmark_type):
    def get_aggregate_score_ci(df, datasets, aggregate_func):
        alg_dict = dict()
        algs = []
        alg_to_idx = dict()
        for i, (alg, df_alg) in enumerate(df.groupby("alg")):
            alg_dict[alg] = df_alg.drop("alg", axis=1).to_numpy()
            algs.append(alg)
            alg_to_idx[alg] = i

        scores, score_cis = get_interval_estimates(alg_dict, aggregate_func, reps=50000) 

        results_dict = {
            "iqm": np.full((len(alg_to_idx), len(datasets)), -1.0),
            "lower_ci": np.full((len(alg_to_idx), len(datasets)), -1.0),
            "upper_ci": np.full((len(alg_to_idx), len(datasets)), -1.0),
        }

        for alg in scores:
            alg_idx = alg_to_idx[alg]
            for i, dataset in enumerate(datasets):
                results_dict["iqm"][alg_idx, i] = scores[alg][i]
                results_dict["lower_ci"][alg_idx, i] = score_cis[alg][0, i]
                results_dict["upper_ci"][alg_idx, i] = score_cis[alg][1, i]

        for dict_type, values in results_dict.items():
            results_dict[dict_type] = pd.DataFrame(values, index=algs, columns=datasets)
        return results_dict

    df = get_benchmark_df(datasets, benchmark_type)
    datasets = df.columns[:-1]
    aggregate_func = lambda x: np.array([aggregate_iqm(x[:, i:i+1]) for i in range(len(datasets))])
    results = get_aggregate_score_ci(df, datasets, aggregate_func)

    df_iqm = get_algorithm_iqm(df)
    erm_iqm = df_iqm.loc["ERM"]
    normalized_df = df.drop("alg", axis=1) / erm_iqm
    normalized_df = pd.concat([normalized_df, df.alg], axis=1)
    aggregate_func = lambda x: np.array([aggregate_iqm(x),])
    agg_results = get_aggregate_score_ci(normalized_df, ["aggregate"], aggregate_func)

    for key in results:
        results[key] = pd.concat([results[key], agg_results[key]], axis=1)

    with open(f"data/iqm_{benchmark_type}.pkl", "wb") as file:
        pickle.dump(results, file)


def get_iqm_diff_bci(datasets, benchmark_type):
    def get_aggregate_score_ci(df, datasets, aggregate_func):
        alg_dict = dict()
        algs = []
        alg_to_idx = dict()
        for i, (alg, df_alg) in enumerate(df.groupby("alg")):
            alg_dict[alg] = df_alg.drop("alg", axis=1).to_numpy()
            algs.append(alg)
            alg_to_idx[alg] = i

        alg_pair_dict = dict()
        for alg, scores in alg_dict.items():
            # if alg == "ERM":
                # continue
            alg_pair_dict[alg] = (scores, alg_dict["ERM"])

        scores, score_cis = get_interval_estimates(alg_pair_dict, aggregate_func, reps=50000)

        results_dict = {
            "iqm": np.full((len(alg_to_idx), len(datasets)), -1.0),
            "lower_ci": np.full((len(alg_to_idx), len(datasets)), -1.0),
            "upper_ci": np.full((len(alg_to_idx), len(datasets)), -1.0),
        }

        for alg in scores:
            alg_idx = alg_to_idx[alg]
            for i, dataset in enumerate(datasets):
                results_dict["iqm"][alg_idx, i] = scores[alg][i]
                results_dict["lower_ci"][alg_idx, i] = score_cis[alg][0, i]
                results_dict["upper_ci"][alg_idx, i] = score_cis[alg][1, i]

        for dict_type, values in results_dict.items():
            results_dict[dict_type] = pd.DataFrame(values, index=algs, columns=datasets)
        return results_dict

    df = get_benchmark_df(datasets, benchmark_type)
    datasets = df.columns[:-1]
    aggregate_func = lambda x, y: np.array([aggregate_iqm_diff(x[:, i:i+1], y[:, i:i+1]) for i in range(len(datasets))])
    results = get_aggregate_score_ci(df, datasets, aggregate_func)

    df_iqm = get_algorithm_iqm(df)
    erm_iqm = df_iqm.loc["ERM"]
    normalized_df = df.drop("alg", axis=1) / erm_iqm
    normalized_df = pd.concat([normalized_df, df.alg], axis=1)
    aggregate_func = lambda x, y: np.array([aggregate_iqm_diff(x, y),])
    agg_results = get_aggregate_score_ci(normalized_df, ["aggregate"], aggregate_func)

    for key in results:
        results[key] = pd.concat([results[key], agg_results[key]], axis=1)

    with open(f"data/iqm_diff_{benchmark_type}.pkl", "wb") as file:
        pickle.dump(results, file)


def plot_loss_heatmaps_posneg(datasets):
    def plot_heatmap(ax, datasets, benchmark_type):
        df = get_benchmark_df(datasets, benchmark_type)
        datasets = df.columns[:-1]
        # with open(f"data/iqm_{benchmark_type}.pkl", "rb") as file:
        with open(f"data/iqm_diff_{benchmark_type}.pkl", "rb") as file:
            iqm_bci_dict = pickle.load(file)
            for key in iqm_bci_dict:
                iqm_bci_dict[key] = iqm_bci_dict[key][datasets]

        datasets = df.columns[:-1]
        iqm_df = get_algorithm_iqm(df)

        erm_iqm = iqm_df.loc["ERM"]
        diff_from_baseline = iqm_df - erm_iqm

        upper_cis = iqm_bci_dict["upper_ci"]
        # erm_is_better = (upper_cis.loc["ERM"] < iqm_df)
        # better_than_erm = (upper_cis < erm_iqm)
        lower_cis = iqm_bci_dict["lower_ci"]
        erm_is_better = (lower_cis > 0)
        better_than_erm = (upper_cis < 0)
        dont_mask = np.logical_or(erm_is_better, better_than_erm)
        diff_from_baseline *= dont_mask.astype(float)

        min_magnitude = np.abs(diff_from_baseline.min(axis=0))
        max_magnitude = diff_from_baseline.max(axis=0)
        diff_from_baseline[diff_from_baseline > 0] /= max_magnitude
        diff_from_baseline[diff_from_baseline < 0] /= min_magnitude

        # Make a new colormap that blends to white at center
        coolwarm = plt.get_cmap('coolwarm')
        colors = coolwarm(np.linspace(0, 1, 256))
        mid = 128
        colors[mid] = [1, 1, 1, 1]  # Set the center color (zero) to white
        custom_cmap = LinearSegmentedColormap.from_list("custom_coolwarm", colors)

        # Normalize with center at 0
        norm = TwoSlopeNorm(vmin=diff_from_baseline.values.min(),
                            vcenter=0,
                            vmax=diff_from_baseline.values.max())

        ax = sns.heatmap(diff_from_baseline, annot=False, cmap=custom_cmap, norm=norm, cbar=False, square=True, ax=ax)
        # ax = sns.heatmap(diff_from_baseline, annot=False, center=0, cmap='coolwarm', cbar=False, square=True, ax=ax)

        # Place Ticks
        ax.xaxis.tick_top()
        ax.tick_params(top=True, bottom=False)
        ax.set_xticklabels(iqm_df.columns, rotation=45, ha='left')
        ax.set_ylabel("")

        # Add grid
        num_rows, num_cols = diff_from_baseline.shape
        for i in range(num_rows + 1):
            ax.hlines(i, *ax.get_xlim(), color='black', linewidth=0.5)
        for j in range(num_cols + 1):
            ax.vlines(j, *ax.get_ylim(), color='black', linewidth=0.5)

        rect = patches.Rectangle(
            (-.025, -.025),
            num_cols,  
            num_rows,  
            linewidth=1.0,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.set_title(benchmark_type)

    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    benchmark_types = ["global", "split"]
    for i, benchmark_type in enumerate(benchmark_types):
        ax = axes[i]
        plot_heatmap(ax, datasets, benchmark_type)
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig("data/images/iqm_heatmap_posneg.png", bbox_inches="tight")


# Modification of rliable's plot_interval_estimates()
def _plot_interval_estimates(
        point_estimates,
        interval_estimates,
        metric_names,
        algorithms=None,
        colors=None,
        color_palette='colorblind',
        max_ticks=4,
        subfigure_width=3.4,
        row_height=0.37,
        **kwargs
    ):

    if algorithms is None:
        algorithms = list(point_estimates.keys())
    num_metrics = len(point_estimates[algorithms[0]])
    figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))
    h = kwargs.pop('interval_height', 0.6)

    for idx, metric_name in enumerate(metric_names):
        for alg_idx, algorithm in enumerate(algorithms):
            ax = axes[idx] if num_metrics > 1 else axes
            # Plot interval estimates.
            lower, upper = interval_estimates[algorithm][:, idx]
            ax.barh(
                y=alg_idx,
                width=upper - lower,
                height=h,
                left=lower,
                color=colors[algorithm],
                alpha=0.75,
                label=algorithm)
            # Plot point estimates.
            ax.vlines(
                x=point_estimates[algorithm][idx],
                ymin=alg_idx - (7.5 * h / 16),
                ymax=alg_idx + (6 * h / 16),
                label=algorithm,
                color='k',
                alpha=0.5)

        ax.set_yticks(list(range(len(algorithms))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(algorithms, fontsize='x-large')
        ax.set_title(metric_name, fontsize='xx-large')
        ax.tick_params(axis='both', which='major')
        plot_utils._decorate_axis(ax, ticklabelsize='xx-large', wrect=5)
        ax.spines['left'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.25)
    return fig, axes


# Modification of rliable's plot_probability_of_improvement()
def _plot_probability_of_improvement(
        probability_estimates,
        probability_interval_estimates,
        pair_separator=',',
        ax=None,
        figsize=(4, 3),
        colors=None,
        color_palette='colorblind',
        alpha=0.75,
        xticks=None,
        xlabel='P(X > Y)',
        left_ylabel='Algorithm X',
        right_ylabel='Algorithm Y',
        algorithms=None,
        **kwargs
    ):

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if not colors:
        colors = sns.color_palette(color_palette, n_colors=len(probability_estimates))
    h = kwargs.pop('interval_height', 0.6)
    wrect = kwargs.pop('wrect', 5)
    ticklabelsize = kwargs.pop('ticklabelsize', 'xx-large')
    labelsize = kwargs.pop('labelsize', 'xx-large')
    ylabel_x_coordinate = kwargs.pop('ylabel_x_coordinate', 0.2)

    twin_ax = ax.twinx()
    all_algorithm_x = []

    # Main plotting code
    if algorithms is None:
        algorithms = list(probability_estimates.keys())

    for idx, algorithm_pair in enumerate(algorithms):
        prob = probability_estimates[algorithm_pair]
        lower, upper = probability_interval_estimates[algorithm_pair]
        algorithm_x, algorithm_y = algorithm_pair.split(pair_separator)
        all_algorithm_x.append(algorithm_x)

        ax.barh(
            y=idx,
            width=upper - lower,
            height=h,
            left=lower,
            color=colors[idx],
            alpha=alpha,
            label=algorithm_x)
        twin_ax.barh(
            y=idx,
            width=upper - lower,
            height=h,
            left=lower,
            color=colors[idx],
            alpha=0.0,
            label=algorithm_y)
        ax.vlines(
            x=prob,
            ymin=idx - 7.5 * h / 16,
            ymax=idx + (6 * h / 16),
            color='k',
            alpha=min(alpha + 0.1, 1.0))

    # Beautify plots
    yticks = range(len(probability_estimates))
    ax = plot_utils._annotate_and_decorate_axis(
        ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticks,
        xlabel=xlabel,
        ylabel=left_ylabel,
        wrect=wrect,
        ticklabelsize=ticklabelsize,
        labelsize=labelsize,
        **kwargs
    )
    twin_ax = plot_utils._annotate_and_decorate_axis(
        twin_ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticks,
        xlabel=xlabel,
        # ylabel=right_ylabel,
        ylabel="",
        wrect=wrect,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        grid_alpha=0.0,
        **kwargs
    )
    ax.set_yticklabels(all_algorithm_x, fontsize='x-large')
    ax.set_ylabel(
        left_ylabel,
        fontweight='bold',
        rotation='horizontal',
        va='bottom',
        fontsize=labelsize
    )
    twin_ax.set_yticklabels([], fontsize=ticklabelsize)
    ax.set_yticklabels(all_algorithm_x, fontsize=ticklabelsize)
    ax.tick_params(axis='both', which='major')
    twin_ax.tick_params(axis='both', which='major')
    ax.spines['left'].set_visible(False)
    twin_ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_coords(-ylabel_x_coordinate, 1.0)
    twin_ax.yaxis.set_label_coords(1 + 0.7 * ylabel_x_coordinate, 1.0)
    ax.grid(False, axis='y')
    twin_ax.grid(False, axis='y')

    return ax


def plot_iqm_cis():
    def get_score_cis(iqm_results, dataset):
        scores = iqm_results["iqm"][dataset]
        score_upper_cis = iqm_results["upper_ci"][dataset]
        score_lower_cis = iqm_results["lower_ci"][dataset]

        score_dict = dict()
        score_ci_dict = dict()
        for alg, val in scores.items():
            score_dict[alg] = np.array([val])
            score_ci_dict[alg] = np.array([[score_lower_cis[alg]], [score_upper_cis[alg]]])
        
        return score_dict, score_ci_dict

    with open(f"data/iqm_global.pkl", "rb") as file:
        iqm_global = pickle.load(file)
    with open(f"data/iqm_split.pkl", "rb") as file:
        iqm_split = pickle.load(file)

    for dataset in iqm_global["iqm"]:
        global_aggregate_scores, global_aggregate_score_cis = get_score_cis(iqm_global, dataset)
        split_aggregate_scores, split_aggregate_score_cis = get_score_cis(iqm_split, dataset)

        algorithms = list(global_aggregate_scores.keys())
        algorithms.pop(algorithms.index("ERM"))
        algorithms = np.sort(algorithms)[::-1]
        algorithms = np.append(algorithms, ["ERM"])

        dataset_scores = dict()
        dataset_score_cis = dict()
        for alg in global_aggregate_scores:
            dataset_scores[alg] = np.concatenate((global_aggregate_scores[alg], split_aggregate_scores[alg]), axis=0)
            dataset_score_cis[alg] = np.concatenate((global_aggregate_score_cis[alg], split_aggregate_score_cis[alg]), axis=1)

        fig, axes = _plot_interval_estimates(
            dataset_scores, dataset_score_cis,
            metric_names=['global', 'split'],
            xlabel='',
            algorithms=algorithms,
            max_ticks=5,
            subfigure_width=6.0)

        plt.tight_layout()
        plt.savefig(f"data/images/{dataset}_iqm.png", bbox_inches="tight")
        plt.close()


def plot_mann_whit_u_cis():
    def get_score_cis(u_results, dataset):
        scores = u_results["u"][dataset].drop("ERM")
        score_upper_cis = u_results["upper_ci"][dataset].drop("ERM")
        score_lower_cis = u_results["lower_ci"][dataset].drop("ERM")

        score_dict = dict()
        score_ci_dict = dict()
        for alg, val in scores.items():
            score_dict[f"{alg},ERM"] = np.array([val])
            score_ci_dict[f"{alg},ERM"] = np.array([[score_lower_cis[alg]], [score_upper_cis[alg]]])
        
        return score_dict, score_ci_dict

    with open(f"data/mann_whit_global.pkl", "rb") as file:
        u_global = pickle.load(file)
    with open(f"data/mann_whit_split.pkl", "rb") as file:
        u_split = pickle.load(file)

    for dataset in u_global["u"]:
        global_aggregate_scores, global_aggregate_score_cis = get_score_cis(u_global, dataset)
        split_aggregate_scores, split_aggregate_score_cis = get_score_cis(u_split, dataset)

        algorithms = list(global_aggregate_scores.keys())
        algorithms = np.sort(algorithms)[::-1]

        fig, axes = plt.subplots(1, 2, figsize=(12.15, 6.9))
        for i, (scores, score_cis) in enumerate(zip([global_aggregate_scores, split_aggregate_scores], [global_aggregate_score_cis, split_aggregate_score_cis])):
            ax = axes[i]
        
            _plot_probability_of_improvement(scores, score_cis, ax=ax, xlabel="P( X < ERM )", algorithms=algorithms)
            if i == 1:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.set_title("split", fontsize='xx-large')
            else:
                ax.set_title("global", fontsize='xx-large')

        plt.tight_layout()
        plt.savefig(f"data/images/{dataset}_poi.png", bbox_inches="tight")
        plt.close()


def get_benchmark_df(datasets, benchmark_type, pretty_names=True):
    dataset_dfs = []
    for dataset in datasets:
        data_dir = f"data/benchmark/{benchmark_type}/{dataset}"
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


def get_algorithm_iqm(df: pd.DataFrame):
    return df.groupby('alg').apply(lambda group: group.apply(lambda x: trim_mean(x, proportiontocut=.25)))


def aggregate_iqm_diff(scores_x, scores_y):
    iqm_x = scipy.stats.trim_mean(scores_x, proportiontocut=0.25, axis=None)
    iqm_y = scipy.stats.trim_mean(scores_y, proportiontocut=0.25, axis=None)
    return iqm_x - iqm_y


def plot_iqm_diff_cis():
    def get_score_cis(iqm_diff_results, dataset):
        scores = iqm_diff_results["iqm"][dataset].drop("ERM")
        score_upper_cis = iqm_diff_results["upper_ci"][dataset].drop("ERM")
        score_lower_cis = iqm_diff_results["lower_ci"][dataset].drop("ERM")

        score_dict = dict()
        score_ci_dict = dict()
        for alg, val in scores.items():
            score_dict[f"{alg},ERM"] = np.array([val])
            score_ci_dict[f"{alg},ERM"] = np.array([[score_lower_cis[alg]], [score_upper_cis[alg]]])
        
        return score_dict, score_ci_dict

    with open(f"data/iqm_diff_global.pkl", "rb") as file:
        iqm_diff_global = pickle.load(file)
    with open(f"data/iqm_diff_split.pkl", "rb") as file:
        iqm_diff_split = pickle.load(file)

    for dataset in iqm_diff_global["iqm"]:
        global_aggregate_scores, global_aggregate_score_cis = get_score_cis(iqm_diff_global, dataset)
        split_aggregate_scores, split_aggregate_score_cis = get_score_cis(iqm_diff_split, dataset)

        algorithms = list(global_aggregate_scores.keys())
        algorithms = np.sort(algorithms)[::-1]

        fig, axes = plt.subplots(1, 2, figsize=(12.15, 6.9))
        for i, (scores, score_cis) in enumerate(zip([global_aggregate_scores, split_aggregate_scores], [global_aggregate_score_cis, split_aggregate_score_cis])):
            ax = axes[i]
        
            _plot_probability_of_improvement(scores, score_cis, ax=ax, xlabel="IQM(X) - IQM(ERM)", algorithms=algorithms)
            if i == 1:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.set_title("split", fontsize='xx-large')
            else:
                ax.set_title("global", fontsize='xx-large')

        plt.tight_layout()
        plt.savefig(f"data/images/{dataset}_iqm_diff.png", bbox_inches="tight")
        plt.close()


def print_aggregate_tables():
    stat_types = ["iqm", "iqm_diff", "mann_whit"]
    stat_names = ["iqm", "iqm", "u"]

    algorithms = None
    for stat_type, stat_name in zip(stat_types, stat_names):
        with open(f"data/{stat_type}_global.pkl", "rb") as file:
            stat_global = pickle.load(file)
        with open(f"data/{stat_type}_split.pkl", "rb") as file:
            stat_split = pickle.load(file)

        if algorithms is None:
            algorithms = stat_global[stat_name]["aggregate"].index
            algorithms = list(algorithms)
            algorithms.pop(algorithms.index("ERM"))
            algorithms = np.sort(algorithms)
            algorithms = np.append(["ERM"], algorithms)

        g_stat = stat_global[stat_name]["aggregate"]
        g_lci = stat_global["lower_ci"]["aggregate"]
        g_uci = stat_global["upper_ci"]["aggregate"]
        s_stat = stat_split[stat_name]["aggregate"]
        s_lci = stat_split["lower_ci"]["aggregate"]
        s_uci = stat_split["upper_ci"]["aggregate"]

        print(f"Table: {stat_type}")
        print(f"Alg & Lower CI & {stat_name} & Upper CI & Lower CI & {stat_name} & Upper CI \\\\")
        for alg in algorithms:
            print(f"{alg} & {g_lci[alg]:.4f} & {g_stat[alg]:.4f} & {g_uci[alg]:.4f} & {s_lci[alg]:.4f} & {s_stat[alg]:.4f} & {s_uci[alg]:.4f} \\\\")
        print("\n\n")



def main():
    args = get_args()

    os.makedirs("data/images/", exist_ok=True)

    get_iqm_bci(args.dataset_type, "global")
    get_iqm_bci(args.dataset_type, "split")
    get_iqm_diff_bci(args.dataset_type, "global")
    get_iqm_diff_bci(args.dataset_type, "split")
    get_mann_whitney_u_bci(args.dataset_type, "global")
    get_mann_whitney_u_bci(args.dataset_type, "split")

    print("Getting aggregate tables\n")
    print_aggregate_tables()

    plot_iqm_cis()
    plot_iqm_diff_cis()
    plot_mann_whit_u_cis()
    plot_loss_heatmaps_posneg(args.dataset_type)

    for benchmark_type in ["global", "split"]:
        df = get_benchmark_df(args.dataset_type, benchmark_type)
        iqm_df = df.groupby("alg").max()
        for alg in iqm_df.index:
            if alg == "ERM":
                continue
            stat, p = scipy.stats.wilcoxon(iqm_df.loc[alg], iqm_df.loc["ERM"], alternative="less")
            if p < .05:
                print(f"{alg}'s max loss is usually less than ERM with a p-value={p} on {benchmark_type} hyperparameter selection")


if __name__=="__main__":
    main()