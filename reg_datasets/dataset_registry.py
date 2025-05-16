from dataclasses import dataclass
from typing import Callable


@dataclass
class DatasetInfo:
    get_dataloaders: Callable
    loss_fn: Callable
    max_steps: int
    log_interval: int
    val_interval: int
    batch_size: int
    max_grad_norm: int
    lr: int
    input_shape: tuple[int]
    n_outputs: int
    default_test_envs: list[int]
    default_test_envs: list[int]


DATASETS: dict[str, DatasetInfo] = dict()

DATASET_DIR = "data/datasets"


def register_dataset(name, get_dataloaders, loss_fn, max_steps, log_interval, val_interval, batch_size, max_grad_norm, lr, input_shape, n_outputs, default_test_envs):
    if name in DATASETS:
        raise ValueError(f"{name} dataset has already been registered")

    if default_test_envs is None:
        print(f"Warning, you need to implement default_test_envs for {name}. Using test_envs=[0]")
        default_test_envs = [0]

    dataset_info = DatasetInfo(
        get_dataloaders=get_dataloaders, 
        loss_fn=loss_fn,
        max_steps=max_steps,
        log_interval=log_interval,
        val_interval=val_interval,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        lr=lr,
        input_shape=input_shape,
        n_outputs=n_outputs,
        default_test_envs=default_test_envs,
    )

    DATASETS[name] = dataset_info
