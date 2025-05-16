import numpy as np
import pandas as pd

from reg_datasets.dataset_registry import register_dataset
from reg_datasets.utils import get_single_serve_dataloader, train_test_domain_split, train_val_data_split, InfiniteFastDataLoader, mse_loss


DATASET_DIR = f"data/datasets"


def get_dataset(n_inv_features, n_samples, hard=False):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def silu(x):
        return x * sigmoid(x)

    def feat_net(x):
        return W3 @ (silu (W2 @ silu(W1 @ x.T)))

    def out_net(x):
        return (q.T @ silu(x)).T

    rng = np.random.default_rng(3)
    W1 = rng.standard_normal((n_inv_features, n_inv_features))
    W2 = rng.standard_normal((n_inv_features, n_inv_features))
    W3 = rng.standard_normal((n_inv_features, n_inv_features))
    q = rng.standard_normal((n_inv_features, 1)) / n_inv_features

    xz_correlations = np.array([.1, .3, .5, .7, .9])
    if hard:
        xz_correlations = np.array([.1, .75, .8, .85, .9])

    xz_stds = np.sqrt(1 / xz_correlations**2 - 1)

    data_envs = []
    for xz_std in xz_stds:
        a = rng.uniform(-2, 2, size=(n_samples, n_inv_features))
        x = a + rng.normal(0, .01, size=(n_samples, n_inv_features))
        feats = feat_net(x)
        z = feats.T + rng.normal(0, xz_std, size=(n_samples, n_inv_features))
        z /= np.std(z)

        y_true = out_net(feats) 
        y_true = y_true - np.mean(y_true)
        y_true /= np.std(y_true) 
        y = y_true + rng.normal(0, .001, size=(n_samples, 1))

        data_envs.append(np.concatenate((x, z, y), axis=1))

    return data_envs


def save_dataset():
    n_inv_features = 8
    n_samples = 4096
    cols = [f"x{i+1}" for i in range(n_inv_features)] + [f"z{i+1}" for i in range(n_inv_features)] + ["y"]

    for hard_version in [False, True]:
        dfs = []
        x_envs = get_dataset(n_inv_features, n_samples, hard=hard_version)
        for i, xs in enumerate(x_envs):
            df = pd.DataFrame(xs, columns=cols)
            df.insert(0, "domain", i)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        filename = "nonlin"
        if hard_version:
            filename += "_hard"
        df.to_csv(f"{DATASET_DIR}/{filename}.csv", index=False)


def load_dataset(hard):
    filename = "nonlin"
    if hard:
        filename += "_hard"
    df = pd.read_csv(f"{DATASET_DIR}/{filename}.csv", index_col="domain")
    data_envs = []
    for _, group in df.groupby("domain"):
        data_envs.append(group.to_numpy())
    return data_envs


def get_nonlin_dataloader_fn(n_inv_features, dataset_fn, hard):
    def get_nonlin_dataloaders(batch_size, device, seed, model_selection_type, test_domains, val_domains, n_val_domains, hparams, n_samples=4096, ):
        rng = np.random.default_rng(seed)
        data_envs = load_dataset(hard)

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
        
    return get_nonlin_dataloaders


register_dataset(
    name="nonlin",
    get_dataloaders=get_nonlin_dataloader_fn(n_inv_features=8, dataset_fn=get_dataset, hard=False),
    loss_fn=mse_loss,
    max_steps=40000,
    log_interval=50,
    val_interval=250,
    batch_size=512,
    max_grad_norm=20,
    lr=1e-3,
    input_shape=(16,),
    n_outputs=1,
    default_test_envs=[0],
)


register_dataset(
    name="nonlin_hard",
    get_dataloaders=get_nonlin_dataloader_fn(n_inv_features=8, dataset_fn=get_dataset, hard=True),
    loss_fn=mse_loss,
    max_steps=40000,
    log_interval=50,
    val_interval=250,
    batch_size=512,
    max_grad_norm=20,
    lr=1e-3,
    input_shape=(16,),
    n_outputs=1,
    default_test_envs=[0],
)