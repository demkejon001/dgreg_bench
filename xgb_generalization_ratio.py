import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from train import set_seed
from reg_datasets.utils import train_test_domain_split, train_val_data_split
from reg_datasets.bike import load_dataset as load_bike_dataset
from reg_datasets.room_occupancy import load_dataset as load_room_dataset
from reg_datasets.income import load_dataset as load_income_dataset
from reg_datasets.poverty import load_dataset as load_poverty_dataset
from reg_datasets.taxi import load_dataset as load_taxi_dataset
from reg_datasets.us_accidents import load_dataset as load_accident_dataset


def split_data(data_envs, test_domains, rng, model_selection_type, model_type):
    def get_xy(data_envs):
        if model_type == "svm":
            new_data_envs = [env[:1000] for env in data_envs]
            data = np.concatenate(new_data_envs, axis=0)
        else:
            data = np.concatenate(data_envs, axis=0)
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    if model_selection_type == "training_domain":
        train_envs, test_envs = train_test_domain_split(data_envs, test_domains)
        train_envs, val_envs = train_val_data_split(train_envs, .2, rng)
    elif model_selection_type == "discrepancy":
        train_envs, test_envs = train_test_domain_split(data_envs, test_domains)
        test_train_envs, test_envs = train_val_data_split(test_envs, .2, rng)
        train_envs = train_envs + test_train_envs
        train_envs, val_envs = train_val_data_split(train_envs, .2, rng)

    x, y = get_xy(train_envs)
    val_x, val_y = get_xy(val_envs)
    test_x, test_y = get_xy(test_envs)
    return x, y, val_x, val_y, test_x, test_y


def get_dataset_statistics(data_envs, model_type, seed=42):
    if model_type == "xgb":
        model = xgb.XGBRegressor(objective="reg:squarederror")
    elif model_type == "lin":
        model = LinearRegression()
    elif model_type == "svm":
        model = SVR()
    else:
        raise ValueError(f"Unrecognized {model_type=}")

    results = []
    for i in range(len(data_envs)):
        test_domains = [i]
        rng = np.random.default_rng(seed)
        x, y, val_x, val_y, test_x, test_y = split_data(data_envs, test_domains, rng, model_selection_type="training_domain", model_type=model_type)
        model.fit(x, y)
        test_pred = model.predict(test_x)
        test_mse = np.sqrt(np.mean((test_pred - test_y)**2))
        train_mse = np.sqrt(np.mean((y - model.predict(x))**2))
        test_r2 = model.score(test_x, test_y)

        rng = np.random.default_rng(seed)
        x, y, val_x, val_y, test_x, test_y = split_data(data_envs, test_domains, rng, model_selection_type="discrepancy", model_type=model_type)
        model.fit(x, y)
        test_pred = model.predict(test_x)
        oracle_test_mse = np.sqrt(np.mean((test_pred - test_y)**2))
        oracle_test_r2 = model.score(test_x, test_y)

        env_results = {
            "gen_ratio": (test_mse / oracle_test_mse), 
            "RMSE_train": train_mse, 
            "RMSE_test": test_mse, 
            "RMSE_oracle": oracle_test_mse, 
            "R2_test": test_r2, 
            "R2_oracle": oracle_test_r2,
        }
        results.append(env_results)

    return results


def get_dataset_avg_statistics(dataset_fn, n_trials = 10):
    trial_results = []
    data_envs = dataset_fn()
    for seed in range(n_trials):
        set_seed(seed)
        results = get_dataset_statistics(data_envs, model_type="xgb", seed=seed)
        trial_results.append(results)

    # env_results = 
    for i in range(n_trials - 1):
        for env in range(len(results)):
            for k, v in trial_results[i][env].items():
                results[env][k] += v
    
    for env in range(len(results)):
        for k in results[env]:
            results[env][k] /= n_trials

    for env in range(len(results)):
        print(f"Env {env} Generalization Ratio: {results[env]['gen_ratio']:.4f}, Train RMSE: {results[env]['RMSE_train']:.4f}, Test RMSE: {results[env]['RMSE_test']:.4f}, Oracle RMSE: {results[env]['RMSE_oracle']:.4f}, Test R2: {results[env]['R2_test']:.4f}, Oracle R2: {results[env]['R2_oracle']:.4f}")


def main():
    def get_room_dataset_fn(dataset_type):
        def room_dataset_fn():
            return load_room_dataset(dataset_type)
        return room_dataset_fn

    dataset_fns = {
        "bike": load_bike_dataset,
        "room_time": get_room_dataset_fn("time"),
        "room_day": get_room_dataset_fn("day"),
        "income": load_income_dataset,
        "poverty": load_poverty_dataset,
        "taxi": load_taxi_dataset,
        "accident": load_accident_dataset,
    }

    for dataset_type, dataset_fn in dataset_fns.items():
        print(f"Dataset {dataset_type}")
        get_dataset_avg_statistics(dataset_fn, n_trials=50)
        print("\n\n")


if __name__=="__main__":
    main()