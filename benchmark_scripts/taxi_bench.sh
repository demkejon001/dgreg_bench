#!/bin/bash

data=taxi
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("2:00:00" "2:00:00" "1:20:00" "1:30:00" "3:00:00" "8:00:00" "0:50:00" "4:00:00" "0:50:00" "1:10:00" "2:00:00" "3:00:00" "2:00:00" "1:30:00" "2:00:00" "2:40:00" "0:50:00" "1:00:00")
HSEEDS=("6 29 54" "2 27 54" "10 27 49" "2 27 50" "9 30 51" "10 37 49" "10 27 49" "5 36 49" "10 27 49" "10 27 49" "2 27 49" "8 32 54" "4 27 49" "2 33 49" "16 31 57" "10 39 49" "11 27 49" "2 27 45")
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

data=taxi
ALGS=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
RUNTIMES=("2:00:00" "2:00:00" "1:20:00" "1:30:00" "3:00:00" "8:00:00" "0:50:00" "4:00:00" "0:50:00" "1:10:00" "2:00:00" "3:00:00" "2:00:00" "1:30:00" "2:00:00" "2:40:00" "0:50:00" "1:00:00")
HSEEDS=("6" "54" "27" "27" "51" "10" "49" "5" "49" "27" "49" "54" "49" "33" "31" "49" "49" "27")
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
