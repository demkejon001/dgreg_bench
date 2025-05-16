#!/bin/bash

data=income
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:20:00" "1:20:00" "0:50:00" "1:00:00" "1:50:00" "4:00:00" "0:30:00" "2:20:00" "0:30:00" "0:40:00" "1:20:00" "1:50:00" "1:20:00" "1:00:00" "1:20:00" "1:40:00" "0:30:00" "0:35:00")
HSEEDS=("2 21 45" "2 34 45" "2 27 45" "2 27 50" "2 27 45" "2 36 51" "2 27 45" "1 27 50" "12 28 47" "2 27 45" "2 27 47" "2 28 45" "2 27 56" "2 27 50" "17 39 42" "2 27 45" "2 27 45" "15 23 45")
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

data=income
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:20:00" "1:20:00" "0:50:00" "1:00:00" "1:50:00" "4:00:00" "0:30:00" "2:20:00" "0:30:00" "0:40:00" "1:20:00" "1:50:00" "1:20:00" "1:00:00" "1:20:00" "1:40:00" "0:30:00" "0:35:00")
HSEEDS=("2" "2" "27" "27" "2" "2" "27" "50" "12" "27" "27" "2" "27" "27" "17" "2" "27" "45")
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
