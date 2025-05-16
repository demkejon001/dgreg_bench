#!/bin/bash

data=accident
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("2:00:00" "2:00:00" "1:20:00" "1:30:00" "3:00:00" "8:00:00" "0:50:00" "4:00:00" "0:50:00" "1:10:00" "2:00:00" "3:00:00" "2:00:00" "1:30:00" "2:00:00" "2:40:00" "0:50:00" "1:00:00")
HSEEDS=("19 21 54" "19 28 59" "19 32 59" "19 30 51" "0 32 59" "19 28 59" "19 32 54" "0 32 51" "13 21 51" "19 28 59" "19 20 59" "11 20 51" "7 32 59" "19 30 51" "16 32 54" "11 21 59" "19 32 59" "19 28 53")
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

data=accident
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("2:00:00" "2:00:00" "1:20:00" "1:30:00" "3:00:00" "8:00:00" "0:50:00" "4:00:00" "0:50:00" "1:10:00" "2:00:00" "3:00:00" "2:00:00" "1:30:00" "2:00:00" "2:40:00" "0:50:00" "1:00:00")
HSEEDS=("21" "19" "19" "19" "32" "19" "19" "32" "21" "19" "59" "20" "59" "19" "32" "59" "32" "19")
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
