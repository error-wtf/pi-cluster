#!/bin/bash
#SBATCH --job-name=pi-64node
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=24
#SBATCH --mem=192G
#SBATCH --time=168:00:00
#SBATCH --output=pi-64node_%j.out
#SBATCH --error=pi-64node_%j.err

# PHYSnet 64-node large-scale pi computation
# module load gcc gmp mpi   # adjust for PHYSnet

echo "=== pi-cluster 64-node run ==="
echo "Nodes: $SLURM_NNODES  Tasks: $SLURM_NTASKS"

cd $SLURM_SUBMIT_DIR

srun ./build/pi-cluster run \
    --digits 100000000000 \
    --backend mpi \
    --output output/pi_large_${SLURM_JOB_ID}.txt \
    --chunk 100 \
    --verbose
