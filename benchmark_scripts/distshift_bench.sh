#!/bin/bash

algs=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
hseeds=("8 33 54" "0 31 47" "8 39 59" "14 20 51" "8 39 46" "13 33 46" "0 39 46" "4 39 46" "0 33 46" "0 39 59" "0 33 59" "0 31 51" "0 39 46" "13 20 51" "0 39 54" "13 39 46" "0 39 46" "0 39 52" )
for i in "${!algs[@]}"; do
  hseed="${hseeds[$i]}"
  alg="${algs[$i]}"
  python run_benchmark.py --alg $alg --data distshift --hparam_seeds $hseed --benchmark_type split --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40
done

algs=(ANDMask CORAL CausIRL_CORAL CausIRL_MMD DAEL EQRM ERM Fish GroupDRO IB_ERM IB_IRM IGA IRM MMD RDM SANDMask SD VREx)
hseeds=("33" "0" "8" "20" "8" "33" "46" "39" "46" "0" "33" "0" "0" "20" "0" "46" "46" "0" )
for i in "${!algs[@]}"; do
  hseed="${hseeds[$i]}"
  alg="${algs[$i]}"
  python run_benchmark.py --alg $alg --data distshift --hparam_seeds $hseed --benchmark_type global --save_best_model --early_stop_start_step 2000 --early_stop_threshold 40
done

