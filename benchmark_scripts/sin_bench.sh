#!/bin/bash

data=sin
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("4:00:00" "4:00:00" "4:00:00" "4:00:00" "6:00:00" "12:00:00" "2:00:00" "8:00:00" "3:00:00" "2:00:00" "4:00:00" "6:00:00" "4:00:00" "4:00:00" "4:00:00" "12:00:00" "2:00:00" "2:30:00")
HSEEDS=("3 22 59" "7 28 50" "7 28 54" "19 21 50" "9 32 59" "7 28 59" "7 21 54" "7 21 55" "19 32 54" "7 28 54" "11 22 59" "11 28 41" "19 21 55" "4 21 50" "19 33 54" "13 21 45" "7 32 54" "4 21 54")
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

data=sin
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("4:00:00" "4:00:00" "4:00:00" "4:00:00" "6:00:00" "12:00:00" "2:00:00" "8:00:00" "3:00:00" "2:00:00" "4:00:00" "6:00:00" "4:00:00" "4:00:00" "4:00:00" "12:00:00" "2:00:00" "2:30:00")
HSEEDS=("3" "28" "54" "21" "32" "59" "21" "7" "54" "28" "59" "28" "21" "21" "33" "21" "32" "54")
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
