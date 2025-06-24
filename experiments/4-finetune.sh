#!/bin/bash

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_INTRA_OP_PARALLELISM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0

cd model || exit 1

nohup mace_run_train --config=../config.yaml --seed=1234 --name=model1 > logs/output1.log &
nohup mace_run_train --config=../config.yaml --seed=2234 --name=model2 > logs/output2.log &

wait
