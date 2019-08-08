#!/usr/bin/env bash
#
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=24000
#SBATCH --account=rkozma
#SBATCH --output=output/dac_%j.out
#SBATCH --cpus-per-task=8

log_dir=$1
seed=${2:-0}
batch_size=${3:-1}
n_neurons=${4:-225}
n_epochs=${5:-5}
time=${6:-100}

cd ../..
echo $log_dir $seed $batch_size $n_neurons $n_epochs $time

python -m minibatch.dac.dac_mnist --gpu \
  --log-dir $log_dir \
  --seed $seed \
  --batch-size $batch_size \
  --n-neurons $n_neurons \
  --n-epochs $n_epochs \
  --time $time

exit
