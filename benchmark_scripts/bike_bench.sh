#!/bin/bash

data=bike
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:00:00" "1:00:00" "0:40:00" "0:50:00" "1:30:00" "4:00:00" "0:25:00" "2:00:00" "0:25:00" "0:35:00" "1:00:00" "1:30:00" "1:00:00" "0:50:00" "1:00:00" "2:00:00" "0:25:00" "0:30:00")
HSEEDS=("0 30 54" "12 21 54" "12 34 54" "2 21 51" "19 29 54" "19 34 59" "12 34 54" "19 37 54" "12 21 47" "9 28 47" "9 21 47" "13 28 47" "0 23 44" "14 27 51" "7 27 46" "1 21 45" "12 34 47" "0 21 46")
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

data=bike
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:00:00" "1:00:00" "0:40:00" "0:50:00" "1:30:00" "4:00:00" "0:25:00" "2:00:00" "0:25:00" "0:35:00" "1:00:00" "1:30:00" "1:00:00" "0:50:00" "1:00:00" "2:00:00" "0:25:00" "0:30:00")
HSEEDS=("0" "54" "12" "21" "54" "19" "12" "54" "12" "9" "9" "47" "44" "51" "27" "45" "34" "0")
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
