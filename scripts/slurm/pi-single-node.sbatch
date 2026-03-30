#!/bin/bash
#SBATCH --job-name=pi-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=pi-single_%j.out
#SBATCH --error=pi-single_%j.err

# PHYSnet single-node pi computation
# Adjust partition/modules to match your PHYSnet environment

# module load gcc gmp   # uncomment and adjust for PHYSnet
# module load cuda      # uncomment if GPU needed

echo "=== pi-cluster single-node run ==="
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Scratch: ${SLURM_TMPDIR:-/tmp}"

cd $SLURM_SUBMIT_DIR

# Detect hardware
./build/pi-cluster detect --json > reports/detect_${SLURM_JOB_ID}.json

# Run benchmark
./build/pi-cluster bench --json > reports/bench_${SLURM_JOB_ID}.json

# Compute pi (adjust digits as needed)
./build/pi-cluster run \
    --digits 1000000 \
    --backend cpu \
    --output output/pi_${SLURM_JOB_ID}.txt \
    --chunk 500 \
    --verbose

echo "=== Done ==="
