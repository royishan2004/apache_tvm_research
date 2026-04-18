#!/bin/bash

echo "Setting performance governor"
sudo cpupower frequency-set -g performance

#echo "Disabling turbo"
#echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

echo "Setting min and max frequency"
sudo cpupower frequency-set -d 3.0GHz
sudo cpupower frequency-set -u 3.0GHz

echo "Setting OpenMP environment variables"
#export OMP_NUM_THREADS=12
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread
export OMP_PROC_BIND=false

echo "Setting number of TVM Threads"
export TVM_NUM_THREADS=12

echo "Benchmark environment ready"

echo "Ensuring taskset across all cores"
exec taskset -c 0-11 bash