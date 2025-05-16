import numpy as np
import pandas as pd

from reg_datasets.utils import train_val_data_split, train_test_domain_split, InfiniteFastDataLoader, get_single_serve_dataloader, mse_loss
from reg_datasets.dataset_registry import register_dataset

DATASET_DIR = "data/datasets"


def save_dataset(n_periods=16, n_samples=1024):
    x = np.linspace(0, 2*np.pi*n_periods, n_samples)
    y = np.sin(x)
    x = x / (np.pi*n_periods) - 1  # normalize between -1 and 1

    data = np.stack((x, y), axis=1)
    data_envs = np.split(data, n_periods)
    dfs = []
    for domain, data_env in enumerate(data_envs):
        df = pd.DataFrame(data_env, columns=["x", "y"])
        df.insert(0, "domain", domain)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.to_csv(f"{DATASET_DIR}/sin.csv", index=False)


def get_dataset():
    data_envs = []
    df = pd.read_csv(f"{DATASET_DIR}/sin.csv", index_col="domain")
    for _, group in df.groupby("domain"):
        data_envs.append(group.to_numpy())
    return data_envs


def get_sin_dataloaders(batch_size, device, seed, model_selection_type, test_domains, val_domains, n_val_domains, hparams, n_periods=16, n_samples=1024, ):
    rng = np.random.default_rng(seed)

    data_envs = get_dataset()

    if model_selection_type == "training_domain":
        train_envs, test_envs = train_test_domain_split(data_envs, test_domains)
        train_envs, val_envs = train_val_data_split(train_envs, .2, rng)
    elif model_selection_type == "discrepancy":
        train_envs, test_envs = train_test_domain_split(data_envs, test_domains)
        test_train_envs, test_envs = train_val_data_split(test_envs, .2, rng)
        train_envs = train_envs + test_train_envs
        train_envs, val_envs = train_val_data_split(train_envs, .2, rng)
    else:
        raise ValueError(f"Unrecognized {model_selection_type=}")

    train_loader = InfiniteFastDataLoader(train_envs, batch_size, device)
    val_loader = get_single_serve_dataloader(val_envs, device)
    test_loader = get_single_serve_dataloader(test_envs, device)

    n_domains = len(train_envs)

    return train_loader, val_loader, test_loader, n_domains


register_dataset(
    name="sin",
    get_dataloaders=get_sin_dataloaders,
    loss_fn=mse_loss,
    max_steps=100000,
    log_interval=100,
    val_interval=1000,
    batch_size=512,
    max_grad_norm=10,
    lr=1e-3,
    input_shape=(1,),
    n_outputs=1,
    default_test_envs=[1, 2, 4, 7, 8, 11, 12, 14],
)
