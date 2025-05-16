#!/bin/bash

data=taxi
ALGS=(ERM SD VREx IB_ERM RDM IRM IB_IRM GroupDRO EQRM CausIRL_CORAL CausIRL_MMD ANDMask SANDMask Fish CORAL MMD IGA DAEL)
RUNTIMES=("0:50:00" "0:50:00" "1:00:00" "1:10:00" "2:00:00" "2:00:00" "2:00:00" "0:50:00" "8:00:00" "1:20:00" "1:30:00" "2:00:00" "2:40:00" "4:00:00" "2:00:00" "1:30:00" "3:00:00" "3:00:00")
for i in ${!ALGS[@]}; do
	alg=${ALGS[$i]}
	time=${RUNTIMES[$i]}
	sbatch --job-name=dgreg_$alg$data \
		--time=$time \
		--gpus=1 \
		--ntasks=10 \
		--mem-per-cpu="1280M" \
		--wrap="OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE python run_random_search.py --alg $alg --data $data --log_interval 250 --early_stop_start_step 2000 --early_stop_threshold 40"
	sleep 1
done
