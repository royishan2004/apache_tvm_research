#!/bin/bash

echo "Setting performance governor"
sudo cpupower frequency-set -g performance

#echo "Disabling turbo"
#echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

echo "Setting min and max frequency"
sudo cpupower frequency-set -d 3.0GHz
sudo cpupower frequency-set -u 3.0GHz

# Ensure virtual environment and Python point to custom-patched TVM (prevents 4-thread throttling bug)
export PYTHONPATH="$(pwd)/tvm/python:$(pwd)"

# Force TVM thread pool to 12 and bind threads
export TVM_NUM_THREADS=12
export TVM_BIND_THREADS=0

# OpenMP: allow 12 threads and explicitly map to all CPUs
export OMP_NUM_THREADS=12
export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}"
export OMP_PROC_BIND=spread

# Marker for Python scripts to verify settings are loaded
export BENCHMARK_SETTINGS_ACTIVE=1

# Allow execution on all cores
taskset -c 0-11 bash