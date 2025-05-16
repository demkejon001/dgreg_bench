#!/bin/bash

data=room_day
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:00:00" "1:00:00" "0:40:00" "0:50:00" "1:30:00" "4:00:00" "0:25:00" "2:00:00" "0:25:00" "0:35:00" "1:00:00" "1:30:00" "1:00:00" "0:50:00" "1:00:00" "2:00:00" "0:25:00" "0:30:00")
HSEEDS=("1 21 41" "9 31 46" "14 31 55" "14 31 41" "15 31 52" "12 31 41" "19 31 41" "11 33 47" "9 24 55" "12 29 43" "4 28 47" "0 28 41" "0 31 55" "14 27 51" "14 26 52" "13 31 45" "19 31 41" "0 24 52")
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

data=room_day
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:00:00" "1:00:00" "0:40:00" "0:50:00" "1:30:00" "4:00:00" "0:25:00" "2:00:00" "0:25:00" "0:35:00" "1:00:00" "1:30:00" "1:00:00" "0:50:00" "1:00:00" "2:00:00" "0:25:00" "0:30:00")
HSEEDS=("41" "31" "14" "14" "52" "31" "31" "11" "24" "29" "28" "41" "31" "14" "26" "31" "31" "52")
for i in ${!ALGS[@]}; do
  alg=${ALGS[$i]}
  time=${RUNTIMES[$i]}
  hseed="${HSEEDS[$i]}"
  sbatch --job-name=bench_$alg$data \
    --time=$time \
    --gpus=1 \
    --ntasks=10 \
    --mem-per-cpu="1024M" \
    --wrap="OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE python run_benchmark.py --alg $alg --data $data --hparam_seeds $hseed --benchmark_type global --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40"
  sleep 1
done
