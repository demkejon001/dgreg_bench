#!/bin/bash

data=poverty
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:20:00" "1:20:00" "0:50:00" "1:00:00" "1:50:00" "4:00:00" "0:30:00" "2:20:00" "0:30:00" "0:40:00" "1:20:00" "1:50:00" "1:20:00" "1:00:00" "1:20:00" "1:40:00" "0:30:00" "0:35:00")
HSEEDS=("14 31 47" "14 27 56" "14 31 47" "1 31 47" "14 31 56" "0 22 46" "18 31 46" "16 24 57" "6 36 46" "1 27 56" "1 27 56" "2 31 47" "6 31 47" "14 31 51" "18 36 46" "14 31 47" "18 31 46" "6 36 47")
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

data=poverty
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:20:00" "1:20:00" "0:50:00" "1:00:00" "1:50:00" "4:00:00" "0:30:00" "2:20:00" "0:30:00" "0:40:00" "1:20:00" "1:50:00" "1:20:00" "1:00:00" "1:20:00" "1:40:00" "0:30:00" "0:35:00")
HSEEDS=("14" "14" "47" "1" "14" "0" "18" "57" "46" "27" "27" "47" "31" "31" "18" "14" "18" "6")
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
