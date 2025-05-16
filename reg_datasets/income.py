import numpy as np
import pandas as pd

from reg_datasets.dataset_registry import register_dataset
from reg_datasets.utils import train_test_domain_split, train_val_data_split, InfiniteFastDataLoader, get_single_serve_dataloader, mse_loss


DATASET_DIR = "data/datasets"


def create_dataset():
    print("This was created with modified Whyshift code")


def load_dataset():
    df = pd.read_csv(f"{DATASET_DIR}/income_age.csv", index_col="domain")
    data_envs = []
    for group_name, group in df.groupby("domain"):
        data_envs.append(group.to_numpy())
    return data_envs


def get_dataloaders(batch_size, device, seed, model_selection_type, test_domains, val_domains, n_val_domains, hparams):
    rng = np.random.default_rng(seed)

    data_envs = load_dataset()

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
    name="income",
    get_dataloaders=get_dataloaders,
    loss_fn=mse_loss,
    max_steps=40000,
    log_interval=50,
    val_interval=250,
    batch_size=512,
    max_grad_norm=20,
    lr=.001,
    input_shape=(76,),
    n_outputs=1,
    default_test_envs=[0],
)
