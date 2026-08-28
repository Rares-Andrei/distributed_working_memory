#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=23:30:00
#SBATCH --partition=batch
#SBATCH --account=vbt
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mail-type=END
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

# How many to run at once:
N_parallel=40

# Build parameter lists
G_list=$(seq -f "%.3f" 0.6   0.01 3)
selectivity_list=$(seq -f "%.3f" 0.5   0.010 1.01)

# Flatten into one queue array of "G selectivity" strings
queue=()
for G in $G_list; do
  for selectivity in $selectivity_list; do
    queue+=("$G $selectivity")
  done
done

# Process in chunks of $N_parallel
total=${#queue[@]}
for start in $(seq 0 $N_parallel $((total-1))); do
  # Determine end index for this batch
  end=$(( start + N_parallel - 1 ))
  if [ "$end" -ge $((total-1)) ]; then
    end=$((total-1))
  fi

  # Launch each job in background
  for i in $(seq "$start" "$end"); do
    # Extract the two parameters
    read G selectivity <<< "${queue[i]}"
    srun -N 1 -n 1 --exact python3 \
         G_selectivity_parameter_sweep_cluster.py \
         "$G" "$selectivity" &
  done

  # Wait for this batch to finish before starting more
  wait
done

# Final wait just in case
wait

echo "All simulations submitted and completed."
