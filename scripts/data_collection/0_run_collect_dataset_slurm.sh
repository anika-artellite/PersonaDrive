#!/bin/bash
#SBATCH --job-name=gen_server
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --output=/YOUR/PATH/results/logs/gen_server_%a_%A.out
#SBATCH --error=/YOUR/PATH/results/logs/gen_server_%a_%A.err
#SBATCH --partition=YOUR_PARTITION

# Submit from the repository root:
#   sbatch scripts/data_collection/0_run_collect_dataset_slurm.sh
# (edit code_root / carla_root in collect_dataset_slurm.py first; the number of
# parallel jobs and the partition are read from max_num_jobs.txt / partition.txt
# next to it and can be changed while it runs).

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

for i in $(seq 1 1); do
  python -u scripts/data_collection/collect_dataset_slurm.py &
done
wait

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"