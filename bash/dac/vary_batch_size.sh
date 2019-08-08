n_neurons=${1:-225}
n_epochs=${2:-5}
time=${3:-100}

for seed in 0 1 2 3 4
do
  for batch_size in 1 2 4 8 16 32 64 128 256 512 1024
  do
    log_dir="jobs/dac/${n_neurons}_${n_epochs}_${time}/${seed}_${batch_size}"
    sbatch submit.sh $log_dir $seed $batch_size $n_neurons $n_epochs $time
  done
done
