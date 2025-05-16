[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

# Domain Generalization Benchmark for Regression
This repository is the official implementation of "Domain Generalization Benchmark for Regression". 
Inspired by [DomainBed](https://github.com/facebookresearch/DomainBed), this codebase contains various domain generalization (DG) algorithms and evaluates them on regression tasks across multiple datasets. 


## Table of Contents
1. [Installation](#installation)
2. [Download the datasets](#download-the-datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Dataset Information](#dataset-information)
7. [References](#references)

## Installation

### Clone the repository

```
git clone https://github.com/demkjon001/dgreg_bench.git
cd [repository-name]
```
### Create and activate a virtual environment (recommended)
```
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```
### Install required dependencies
```
pip install -r requirements.txt
```

## Download the datasets
All datasets utilized in this project are publicly available and come with various licenses. 
Each dataset can be accessed and downloaded from this [Harvard Dataverse Repository](https://dataverse.harvard.edu/dataverse/dgreg_bench). 
Once downloaded, place the datasets in the ```data/datasets/``` folder. 
The training code will automatically use the data from this directory when executed.

For detailed information about each dataset, including its references, licensing terms, and additional context, please refer to the [link](https://dataverse.harvard.edu/dataverse/dgreg_bench) above. 
Alternatively, this information is also provided in the [Dataset Information](#dataset-information) section below.

## Training

Training is broken into two phases: **Hyperparameter Search** and **Benchmarking**.
In all of the cases, the algorithms you can select from are `(ERM, SD, VREx, IB_ERM, RDM, IRM, IB_IRM, GroupDRO, EQRM, CausIRL_CORAL, CausIRL_MMD, ANDMask, SANDMask, Fish, CORAL, MMD, IGA, DAEL)`
and the datasets you choose are `(nonlin, nonlin_hard, sin, distshift, bike, room_day, room_time, income, poverty, taxi, accident)`

<details>
<summary><h3>Random Search</h3></summary>

The hyperparameter search code is found in `random_search.py`. This can be run in three ways:

1. You can run a single random search trial with:
```sh
python random_search.py --alg ERM --data nonlin --log_interval 250 --early_stop_start_step 2000 --early_stop_threshold 40 --hparams_seed 0 --seed 0
```
The `--seed` should remain 0, but `--hparams_seed` can be anywhere from 0-59. 
You should iterate through all possible algorithms and datasets.


2. You can utilize multiprocessing on a single machine with:
```sh
alg=ERM
data=nonlin
python run_random_search.py --alg $alg --data $data --log_interval 250 --early_stop_start_step 2000 --early_stop_threshold 40
```
which will iterate through all 60 `--hparams_seed`, while running 10 experiments in parallel. 


3. If you have a Slurm-capable machine cluster, you can run:
```sh
data=nonlin
ALGS=(ERM SD VREx IB_ERM RDM IRM IB_IRM GroupDRO EQRM CausIRL_CORAL CausIRL_MMD ANDMask SANDMask Fish CORAL MMD IGA DAEL)
RUNTIMES=("0:50:00" "0:50:00" "1:00:00" "1:10:00" "2:00:00" "2:00:00" "2:00:00" "0:50:00" "12:00:00" "2:30:00" "2:30:00" "2:00:00" "2:40:00" "4:00:00" "2:00:00" "1:30:00" "3:00:00" "3:00:00")
for i in ${!ALGS[@]}; do
        alg=${ALGS[$i]}
        time=${RUNTIMES[$i]}
        sbatch --job-name=dgreg_$alg$data \
                --time=$time \
                --gpus=1 \
                --ntasks=10 \
                --mem-per-cpu="1024M" \
                --wrap="OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE python run_random_search.py --alg $alg --data $data --log_interval 250 --early_stop_start_step 2000 --early_stop_threshold 40"
        sleep 1
done
```


</details>

<details>
<summary><h3>Benchmarking</h3></summary>

The benchmarking code is found in `benchmark.py`. 
Once you have completed the 60 runs for all algorithms on a dataset, you can generate a slurm script with the optimal hyperparameters by running:
```sh
python evaluate_random_search.py --data nonlin
```

which will output to the terminal a slurm script, e.g.:

```bash
#!/bin/bash

data=nonlin
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("2:00:00" "2:00:00" "1:20:00" "1:30:00" "3:00:00" "8:00:00" "0:50:00" "4:00:00" "0:50:00" "1:10:00" "2:00:00" "3:00:00" "2:00:00" "1:30:00" "2:00:00" "2:40:00" "0:50:00" "1:00:00")
HSEEDS=("10 24 51" "10 22 51" "10 22 51" "13 24 51" "10 39 51" "10 22 51" "10 22 57" "5 39 51" "5 39 51" "10 22 51" "11 22 51" "10 22 51" "5 22 48" "16 24 51" "11 20 50" "10 22 57" "10 26 51" "5 22 57")
for i in ${!ALGS[@]}; do
  alg=${ALGS[$i]}
  time=${RUNTIMES[$i]}
  hseed="${HSEEDS[$i]}"
  sbatch --job-name=bench_$alg$data \
    --time=$time \
    --gpus=1 \
    --ntasks=10 \
    --mem-per-cpu="1024M" \
    --wrap="OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE python run_benchmark.py --alg $alg --data $data --hparam_seeds $hseed --benchmark_type split --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40"
  sleep 1
done
```

As with `random_search.py`, you can optionally run `benchmark.py` with multiprocessing via:
```sh
# For the split methodology hyperparameters
python run_benchmark.py --alg $alg --data $data --hparams_seeds $hseed --benchmark_type split --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40
# For the global methodology hyperparameters
python run_benchmark.py --alg $alg --data $data --hparams_seeds $hseed --benchmark_type global --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40
```

You can also manually run a single benchmark trial directly (each trial would use the same `--hparams_seed` if you want to use the same hyperparameters, but would use a different `--seed`):
```sh
python benchmark.py --alg ANDMask --data nonlin --hparams_seed 10 --benchmark_type split --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40 --seed 1
```

</details>

## Evaluation

After you have benchmarked all algorithms on all datasets, you can evaluate your results with:
```sh
python evaluate_benchmark.py --data nonlin nonlin_hard sin distshift bike room_day room_time income poverty taxi accident
```

All images (including the ones found in the results section below will be generated).
Computing the (stratified) bootstrap confidence intervals (BCI) is time intensive; taking roughly 5 hours, especially for the Mann-Whitney U-statistic BCIs. 
If you wish to get roughly similar results without waiting for 5 hours, reduce the bootstrap repetition in `get_mann_whitney_u_bci()` to 2000 to reduce the time to roughly 20 minutes.


## Results
![aggregate_poi](https://github.com/user-attachments/assets/5546ef05-effb-41f5-9e18-27b74f2752e7)
![aggregate_iqm](https://github.com/user-attachments/assets/83e7f3d2-7f33-41fd-8a6e-1f2c6d9dc8de)
![aggregate_iqm_diff](https://github.com/user-attachments/assets/013913d5-f6b1-4d59-9e38-acc2d6cd02eb)
![iqm_heatmap_posneg](https://github.com/user-attachments/assets/901f0781-bd51-4c8b-9eb7-3b090ff5aeeb)

## Dataset Information

### Synthetic Nonlinear
* The task is to predict the target variable y based on input features x and noise z, where the relationship between x and y is modeled using a nonlinear function. 
* License: MIT License

### Sin Wave
* The task is to predict the y value returned from sin function given the x value.
* License: CC0 1.0

### Distshift World Modelling
* The task is to predict the next state and reward in a gridworld given the state-action pair.
* Reference: Jan Leike, Miljan Martic, Victoria Krakovna, Pedro A. Ortega, Tom Everitt, Andrew Lefrancq, Laurent Orseau, and Shane Legg. AI safety gridworlds. CoRR, abs/1711.09883, 2017.391
* License: MIT License

### Room Occupancy
* The task is to predict room occupancy counts given sensor data (e.g., temperature, humidity, light, CO2).
* Reference: Singh, A. & Chaudhari, S. (2018). Room Occupancy Estimation. UCI Machine Learning Repository. https://doi.org/10.24432/C5P605.
* License: CC BY 4.0

### Bike
* The task is to predict the number of hourly bike rentals given weather features (e.g., temperature, wind speed, humidity).
* Reference: Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.
* License: CC BY 4.0

### ACS Income
* The task is to predict an individuals income based on U.S Census data.
* Reference: Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. Advances in neural information processing systems, 34, 6478-6490.
* License: MIT License

### ACS Poverty Ratio
* The task is to predict the poverty ratio of individuals based on U.S Census data.
* Reference: Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. Advances in neural information processing systems, 34, 6478-6490.
* License: MIT License

### Taxi
* The task is to predict the total trip duration based on datetime and coordinate data (pickup and dropoff longitude and latitude).
* Reference: Mario Navas. Taxi routes of mexico city, quito and more. https://www.kaggle.com/datasets/mnavas/taxi-routes-for-mexico-city-and-quito, 2017. Kaggle.
* License: CC BY-SA 4.0

### US Accident
* The task is to predict severity of an accident (length of time the accident held up traffic) based on datetime data, weather conditions, and road conditions.
* Reference: Sobhan Moosavi. (2023). US Accidents (2016 - 2023). Kaggle. https://doi.org/10.34740/KAGGLE/DS/199387
* License: CC BY-SA 4.0

## References
[1] Liu, J., Wang, T., Cui, P., & Namkoong, H. (2023). On the need for a language describing distribution shifts: Illustrations on tabular datasets. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.
* We used and modified the [whyshift](https://github.com/namkoong-lab/whyshift/) code to preprocess datasets such as US Accidents, Taxi, and the ACS Datasets.

[2] Gulrajani, I., & Lopez-Paz, D. (2020). In search of lost domain generalization. arXiv. https://arxiv.org/abs/2007.01434
* The [DomainBed](https://github.com/facebookresearch/DomainBed) codebase formed a significant portion of our methodology, and we utilized their algorithms and hyperparameters.

[3] Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. (2021). Deep reinforcement learning at the edge of the statistical precipice. Advances in Neural Information Processing Systems, 34.
* We used and modified the [rliable](https://github.com/google-research/rliable) code to compute our bootstrap confidence intervals and to create pretty plots.
