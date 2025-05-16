#!/bin/bash

data=nonlin_hard
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("2:00:00" "2:00:00" "1:20:00" "1:30:00" "3:00:00" "8:00:00" "0:50:00" "4:00:00" "0:50:00" "1:10:00" "2:00:00" "3:00:00" "2:00:00" "1:30:00" "2:00:00" "2:40:00" "0:50:00" "1:00:00")
HSEEDS=("10 22 51" "10 22 51" "10 22 51" "2 22 51" "10 22 51" "10 22 51" "10 39 51" "5 39 51" "10 39 51" "10 22 51" "5 22 51" "5 22 51" "5 22 40" "2 22 51" "11 20 50" "10 24 57" "10 22 51" "5 30 51")
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

data=nonlin_hard
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("2:00:00" "2:00:00" "1:20:00" "1:30:00" "3:00:00" "8:00:00" "0:50:00" "4:00:00" "0:50:00" "1:10:00" "2:00:00" "3:00:00" "2:00:00" "1:30:00" "2:00:00" "2:40:00" "0:50:00" "1:00:00")
HSEEDS=("22" "22" "22" "22" "22" "10" "10" "39" "10" "22" "22" "51" "22" "22" "50" "10" "10" "5")
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
