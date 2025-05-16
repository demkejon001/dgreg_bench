import pandas as pd
import numpy as np
import pandas as pd
import pickle
import torch
import os

from sklearn.preprocessing import StandardScaler
from reg_datasets.dataset_registry import register_dataset
from reg_datasets.utils import train_test_domain_split, train_val_data_split, InfiniteFastDataLoader, get_single_serve_dataloader, mse_loss


DATASET_DIR="data/datasets"


def create_dataset():
    data_dir = f'{DATASET_DIR}/accident_raw/US_Accidents_March23.csv'
    states = ['CA', 'TX', 'FL', 'OR', 'MN']
    
    raw_X = preprocess(data_dir)
    dfs = []
    for domain, s in enumerate(states):
        data = raw_X[raw_X["State"]==s]
        data = data.drop("State", axis=1)
        data.insert(0, "domain", domain)
        dfs.append(data)
    df = pd.concat(dfs, axis=0)
    df.to_csv(f"{DATASET_DIR}/accident.csv", index=False)
    

def create_dfs():
    data_dir = f'{DATASET_DIR}/accident_raw/US_Accidents_March23.csv'
    states = ['CA', 'TX', 'FL', 'OR', 'MN']
    
    raw_X = preprocess(data_dir)
    data_envs = []
    for s in states:
        data = raw_X[raw_X["State"]==s]
        data = data.drop("State", axis=1)
        data_envs.append(data)
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(f"{DATASET_DIR}/accident_dfs.pkl", "wb") as f:
        pickle.dump(data_envs, f)
    return data_envs


def load_dataset():
    df = pd.read_csv(f"{DATASET_DIR}/accident.csv", index_col="domain")
    data_envs = []
    for _, group in df.groupby("domain"):
        data_envs.append(group.to_numpy())
    return data_envs


def load_dfs():
    with open(f"{DATASET_DIR}/accident_dfs.pkl", "rb") as file:
        data = pickle.load(file)
    return data


def preprocess(dir):
    try:
        data = pd.read_csv(dir)
    except:
        raise FileNotFoundError('File does not exist: {}'.format(dir))
    X = data.copy()
    
    X["Start_Time"] = pd.to_datetime(X["Start_Time"].str.split('.').str[0])
    X["End_Time"] = pd.to_datetime(X["End_Time"].str.split('.').str[0])
    X["Crash_Time"] = (X["End_Time"] - X["Start_Time"]).dt.total_seconds()
    X.loc[X["Crash_Time"] > 86400, "Crash_Time"] = 86400
    X["Crash_Time"] = np.log10(X["Crash_Time"])
    
    # Extract year, month, weekday and day
    X["Year"] = X["Start_Time"].dt.year
    X["Month"] = X["Start_Time"].dt.month
    X["Weekday"] = X["Start_Time"].dt.weekday
    X["Day"] = X["Start_Time"].dt.day
    
    # Extract hour and minute
    X["Hour"] = X["Start_Time"].dt.hour
    X["Minute"] = X["Start_Time"].dt.minute
    
    features_to_drop = ["ID", "Source", "Severity", "Start_Time", "End_Time", "End_Lat", "End_Lng", "Start_Lat", "Start_Lng", "Description", "Street", "County", "Zipcode", "City", "Country", "Timezone", "Airport_Code", "Weather_Timestamp", "Sunrise_Sunset", "Nautical_Twilight", "Astronomical_Twilight"]
    X = X.drop(features_to_drop, axis=1)
    X.drop_duplicates(inplace=True)
    
    X = X[X["Pressure(in)"] != 0]
    X = X[X["Visibility(mi)"] != 0]
    
    X.loc[X["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
    X.loc[X["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
    X.loc[X["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
    X.loc[X["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
    X.loc[X["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
    X.loc[X["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
    X.loc[X["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
    X.loc[X["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
    X.loc[X["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
    X.loc[X["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
    X.loc[X["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan
    
    X.loc[X["Wind_Direction"] == "CALM", "Wind_Direction"] = "Calm"
    X.loc[X["Wind_Direction"] == "VAR", "Wind_Direction"] = "Variable"
    X.loc[X["Wind_Direction"] == "East", "Wind_Direction"] = "E"
    X.loc[X["Wind_Direction"] == "North", "Wind_Direction"] = "N"
    X.loc[X["Wind_Direction"] == "South", "Wind_Direction"] = "S"
    X.loc[X["Wind_Direction"] == "West", "Wind_Direction"] = "W"
    
    features_to_fill = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
    X[features_to_fill] = X[features_to_fill].fillna(X[features_to_fill].mean())
    X.dropna(inplace=True)
    X["Wind_Direction"] = X["Wind_Direction"].map(lambda x : x if len(x) != 3 else x[1:], na_action="ignore")
    
    scaler = StandardScaler()
    features = ['Crash_Time','Temperature(F)','Wind_Chill(F)','Distance(mi)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Year', 'Month','Weekday','Day','Hour','Minute']
    X[features] = scaler.fit_transform(X[features])
    
    categorical_features = ["Wind_Direction", "Weather_Condition", "Civil_Twilight"]
    for cat in categorical_features:
        X[cat] = X[cat].astype("category")
    onehot_cols = categorical_features 
    X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)
    X = X.replace([True, False], [1, 0])
    
    other_columns = [col for col in X.columns if col != "Crash_Time"]
    X = X[other_columns + ["Crash_Time"]]
    return X


class InfiniteFastLazyDataLoader(InfiniteFastDataLoader):
    def __init__(self, data_envs: list[np.ndarray], batch_size, device):
        super().__init__(data_envs, batch_size, device="cpu")
        self.device = device
    
    def __next__(self):
        indices = torch.cat([torch.randint(0, self.n_samples_per_domain[i], size=(self.batch_size_per_domain,)) for i in range(self.n_domains)])
        indices = indices + self.domain_start_indices
        return self.x[indices].to(self.device), self.y[indices].to(self.device)
    

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

    train_loader = InfiniteFastLazyDataLoader(train_envs, batch_size, device)
    val_loader = get_single_serve_dataloader(val_envs, device)
    test_loader = get_single_serve_dataloader(test_envs, device)

    n_domains = len(train_envs)
    return train_loader, val_loader, test_loader, n_domains


register_dataset(
    name="accident",
    get_dataloaders=get_dataloaders,
    loss_fn=mse_loss,
    max_steps=40000,
    log_interval=50,
    val_interval=250,
    batch_size=512,
    max_grad_norm=20,
    lr=.001,
    input_shape=(47,),
    n_outputs=1,
    default_test_envs=[3]
)