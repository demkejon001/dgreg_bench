import numpy as np
import torch


def mse_loss(y_hat, y, reduction="mean"):
    loss = torch.nn.functional.mse_loss(y_hat, y, reduction=reduction)
    return {"loss": loss}


def train_test_domain_split(datasets: list, test_domains: list[int], val_domains = None):
    train_datasets = []
    test_datasets = []
    val_datasets = []

    if val_domains is None:
        val_domains = []
    for i in val_domains:
        if i in test_domains:
            raise ValueError(f"Validation Dataset(s) should not intersect with Test Dataset(s). Change val_envs.")

    n_datasets = len(datasets)
    for i in range(n_datasets):
        if i in test_domains:
            test_datasets.append(datasets[i])
        elif i in val_domains:
            val_datasets.append(datasets[i])
        else:
            train_datasets.append(datasets[i])

    if val_domains:
        return train_datasets, test_datasets, val_datasets
    return train_datasets, test_datasets


def train_test_rand_domain_split(datasets: list, n_test_domains: int, rng: np.random.Generator):
    test_domains = rng.integers(0, len(datasets), size=(int(n_test_domains),))
    return train_test_domain_split(datasets, test_domains)


# Passing in torch.Dataset objects can be expensive
def train_val_data_split(datasets: list, val_size: float, rng: np.random.Generator):
    assert 0.0 < val_size < 1.0

    train_datasets = []
    val_datasets = []

    # Taking min so that validation dataset has same number of samples for each domain/env
    n_val_samples = min([int(len(dataset) * val_size) for dataset in datasets])

    for dataset in datasets:
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        train_indices = indices[n_val_samples:]
        val_indices = indices[:n_val_samples]
        if isinstance(dataset, np.ndarray):
            train_datasets.append(dataset[train_indices])
            val_datasets.append(dataset[val_indices])
        else:
            train_datasets.append([dataset[i] for i in train_indices])
            val_datasets.append([dataset[i] for i in val_indices])

    return train_datasets, val_datasets


def get_single_serve_dataloader(data_envs: list[np.ndarray], device):
    data = np.concat(data_envs, 0)
    x = data[:, :-1]
    y = data[:, -1:]
    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    return [(x, y)]


class InfiniteFastDataLoader:
    def __init__(self, data_envs: list[np.ndarray], batch_size, device):
        data = np.concat(data_envs, 0)
        x = data[:, :-1]
        y = data[:, -1:]

        self.x = torch.from_numpy(x).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)
        self.n_domains = len(data_envs)

        domain_start_idx = 0
        domain_start_indices = []
        n_samples_per_domain = []
        for env in data_envs:
            domain_start_indices.append(domain_start_idx)
            domain_start_idx += len(env)
            n_samples_per_domain.append(len(env))

        self.domain_start_indices = torch.tensor(domain_start_indices).repeat_interleave(batch_size//self.n_domains, 0)

        self.n_samples_per_domain = n_samples_per_domain
        self.batch_size = batch_size
        if (batch_size % self.n_domains) != 0:
            new_batch_size = batch_size - (batch_size % self.n_domains)
            print(f"{batch_size=} doesn't evenly divide with n_domains={self.n_domains}, setting batch_size={new_batch_size}")
            batch_size = new_batch_size
        self.batch_size = batch_size
        self.batch_size_per_domain = self.batch_size // self.n_domains

    def __iter__(self):
        return self 
    
    def __next__(self):
        indices = torch.cat([torch.randint(0, self.n_samples_per_domain[i], size=(self.batch_size_per_domain,)) for i in range(self.n_domains)])
        indices = indices + self.domain_start_indices
        return self.x[indices], self.y[indices]

