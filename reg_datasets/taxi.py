import numpy as np
import pandas as pd

from reg_datasets.utils import train_test_domain_split, train_val_data_split, InfiniteFastDataLoader, get_single_serve_dataloader, mse_loss
from reg_datasets.dataset_registry import register_dataset


DATASET_DIR = "data/datasets"
MAX_TRIP_DURATION = 5900


def filter_dfs(dfs):
    for name, df in dfs.items():
        dfs[name] = df[(df.trip_duration < MAX_TRIP_DURATION) & (df.trip_duration > 0)]
    return dfs


def remove_cols(dfs):
    remove_col_nyc = ['id', 'dropoff_datetime', 'vendor_id', 'passenger_count'] + ["store_and_fwd_flag"]
    remove_col_other = ['id', 'dropoff_datetime', 'vendor_id', 'dist_meters', 'wait_sec', ] + ["store_and_fwd_flag"]
    for name, df in dfs.items():
        if name == "nyc":
            df = df.drop(remove_col_nyc, axis=1)
        else:
            df = df.drop(remove_col_other, axis=1)
        df["city"] = name
        dfs[name] = df
    return dfs


def add_features(dfs):
    def add_cyclic_features(feature, max_val):
        df.insert(0, f"{feature}_cos", np.cos(df[feature] * (2 * np.pi / max_val)))
        df.insert(0, f"{feature}_sin", np.cos(df[feature] * (2 * np.pi / max_val)))
        df.drop([feature], axis=1, inplace=True)

    for name, df in dfs.items():
        # df.drop(['store_and_fwd_flag'], axis=1, inplace=True)
        # df.drop(['vendor_id'], axis=1, inplace=True)
        df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
        # df.drop(['dropoff_datetime'], axis=1, inplace=True) 
        df['month'] = df.pickup_datetime.dt.month
        df['week'] = df.pickup_datetime.dt.isocalendar().week
        df['weekday'] = df.pickup_datetime.dt.weekday
        df['hour'] = df.pickup_datetime.dt.hour
        df['minute'] = df.pickup_datetime.dt.minute
        df['minute_oftheday'] = df['hour'] * 60 + df['minute']
        df.drop(['minute'], axis=1, inplace=True)
        df.drop(['pickup_datetime'], axis=1, inplace=True)

        add_cyclic_features("month", 12)
        add_cyclic_features("week", 7)
        add_cyclic_features("hour", 24)
        add_cyclic_features("minute_oftheday", 24*60)
    return dfs


def scale_features(dfs):
    all_dfs = pd.concat([df for df in dfs.values()])
    all_dfs["trip_duration"] = np.log10(all_dfs["trip_duration"]) / np.log10(MAX_TRIP_DURATION)
    all_dfs['pickup_longitude'] = all_dfs['pickup_longitude'] / 180.0
    all_dfs['dropoff_longitude'] = all_dfs['dropoff_longitude'] / 180.0
    all_dfs['pickup_latitude'] = all_dfs['pickup_latitude'] / 90.0
    all_dfs['dropoff_latitude'] = all_dfs['dropoff_latitude'] / 90.0
    dfs = dict()
    for name, df in all_dfs.groupby("city"):
        dfs[name] = df
    return dfs


def create_dataset():
    mex_df = pd.read_csv(f"data/datasets/taxi_raw/mex_clean.csv")
    bog_df = pd.read_csv(f"data/datasets/taxi_raw/bog_clean.csv")
    uio_df = pd.read_csv(f"data/datasets/taxi_raw/uio_clean.csv")
    nyc_df = pd.read_csv(f"data/datasets/taxi_raw/train.csv")

    df_dict = {
        "mex": mex_df,
        "bog": bog_df,
        "uio": uio_df,
        "nyc": nyc_df,
    }

    df_dict = add_features(df_dict)
    df_dict = filter_dfs(df_dict)
    df_dict = remove_cols(df_dict)
    df_dict = scale_features(df_dict)

    cities = ["nyc", "mex", "bog", "uio"]
    # envs = [(df_dict[city].drop(["city", "weekday"], axis=1)).to_numpy().astype(float) for city in cities]

    dfs = []
    for domain, city in enumerate(cities):
        df = df_dict[city].drop(["city", "weekday"], axis=1)
        df.insert(0, "domain", domain)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.to_csv(f"{DATASET_DIR}/taxi.csv", index=False)


def load_dataset():
    df = pd.read_csv(f"{DATASET_DIR}/taxi.csv", index_col="domain")
    envs = []
    for _, group in df.groupby("domain"):
        envs.append(group.to_numpy())
    return envs


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
    name="taxi",
    get_dataloaders=get_dataloaders,
    loss_fn=mse_loss,
    max_steps=40000,
    log_interval=50,
    val_interval=250,
    batch_size=256,
    max_grad_norm=20,
    lr=.001,
    input_shape=(12,),
    n_outputs=1,
    default_test_envs=[0],
)
