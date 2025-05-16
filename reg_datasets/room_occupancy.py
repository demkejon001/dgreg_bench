from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from reg_datasets.dataset_registry import register_dataset
from reg_datasets.utils import train_test_domain_split, train_val_data_split, InfiniteFastDataLoader, get_single_serve_dataloader, mse_loss


DATASET_DIR="data/datasets"


def get_original_room_data():
    room_occupancy_estimation = fetch_ucirepo(id=864) 
    X = room_occupancy_estimation.data.features 
    y = room_occupancy_estimation.data.targets 
    return X.join(y)


def preprocess_room_data(data):
    filtered_data = data.drop(["Date", "Time"], axis="columns")
    data["Hour"] = pd.to_timedelta(data.Time).dt.total_seconds() / 3600
    filtered_data.insert(0, "Hour_cos", np.cos(data["Hour"] * (2 * np.pi / 24.0)))
    filtered_data.insert(0, "Hour_sin", np.sin(data["Hour"] * (2 * np.pi / 24.0)))

    filtered_data[filtered_data.columns] = MinMaxScaler().fit_transform(filtered_data)
    return filtered_data


def create_date_dataset():
    data = get_original_room_data()
    filtered_data = preprocess_room_data(data)

    dfs = []
    for domain, date in enumerate(data.Date.unique()):
        df = pd.DataFrame(filtered_data.loc[data.Date == date].to_numpy(), columns=filtered_data.columns)
        df.insert(0, "domain", domain)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.to_csv(f"{DATASET_DIR}/room_day.csv")
    print(df.shape)


def create_time_dataset():
    data = get_original_room_data()
    filtered_data = preprocess_room_data(data)

    dfs = []
    steps = 4
    for domain, i in enumerate(range(0, 24, steps)):
        df = pd.DataFrame(filtered_data.loc[(i < data.Hour) & (data.Hour < i + steps)], columns=filtered_data.columns)
        df.insert(0, "domain", domain)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.to_csv(f"{DATASET_DIR}/room_time.csv")
    print(df.shape)


def create_datasets():
    create_date_dataset()
    create_time_dataset()
    

def load_dataset(dataset_type):
    if dataset_type not in ["day", "time"]:
        raise ValueError(f"Unrecognized {dataset_type=}") 

    df = pd.read_csv(f"{DATASET_DIR}/room_{dataset_type}.csv", index_col="domain")
    data_envs = []
    for group_name, group in df.groupby("domain"):
        data_envs.append(group.to_numpy())
    return data_envs


def get_dataloader_fn(dataset_type):
    def get_dataloaders(batch_size, device, seed, model_selection_type, test_domains, val_domains, n_val_domains, hparams):
        rng = np.random.default_rng(seed)

        data_envs = load_dataset(dataset_type)

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
    return get_dataloaders
            

register_dataset(
    name="room_time",
    get_dataloaders=get_dataloader_fn("time"),
    loss_fn=mse_loss,
    max_steps=20000,
    log_interval=50,
    val_interval=250,
    batch_size=256,
    max_grad_norm=20,
    lr=.001,
    input_shape=(18,),
    n_outputs=1,
    default_test_envs=[4],
)

register_dataset(
    name="room_day",
    get_dataloaders=get_dataloader_fn("day"),
    loss_fn=mse_loss,
    max_steps=20000,
    log_interval=50,
    val_interval=250,
    batch_size=256,
    max_grad_norm=20,
    lr=.001,
    input_shape=(18,),
    n_outputs=1,
    default_test_envs=[1],
)