#!/bin/bash

data=room_time
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:00:00" "1:00:00" "0:40:00" "0:50:00" "1:30:00" "4:00:00" "0:25:00" "2:00:00" "0:25:00" "0:35:00" "1:00:00" "1:30:00" "1:00:00" "0:50:00" "1:00:00" "2:00:00" "0:25:00" "0:30:00")
HSEEDS=("13 36 46" "6 30 46" "0 34 47" "1 20 50" "6 32 44" "0 30 47" "0 33 44" "0 37 46" "0 37 46" "6 30 46" "0 20 47" "0 31 59" "12 23 47" "14 30 51" "19 27 44" "12 28 54" "3 37 44" "0 32 46")
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

data=room_time
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("1:00:00" "1:00:00" "0:40:00" "0:50:00" "1:30:00" "4:00:00" "0:25:00" "2:00:00" "0:25:00" "0:35:00" "1:00:00" "1:30:00" "1:00:00" "0:50:00" "1:00:00" "2:00:00" "0:25:00" "0:30:00")
HSEEDS=("13" "6" "0" "20" "32" "47" "0" "46" "0" "30" "47" "0" "47" "51" "27" "54" "44" "46")
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
