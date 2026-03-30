#!/bin/bash
#SBATCH --job-name=pi-8node
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=24
#SBATCH --mem=192G
#SBATCH --time=120:00:00
#SBATCH --output=pi-8node_%j.out
#SBATCH --error=pi-8node_%j.err

# PHYSnet 8-node MPI pi computation
# module load gcc gmp mpi   # adjust for PHYSnet

echo "=== pi-cluster 8-node MPI run ==="
echo "Nodes: $SLURM_NNODES  Tasks: $SLURM_NTASKS"
echo "Nodelist: $SLURM_JOB_NODELIST"

cd $SLURM_SUBMIT_DIR

srun ./build/pi-cluster run \
    --digits 1000000000 \
    --backend mpi \
    --output output/pi_mpi_${SLURM_JOB_ID}.txt \
    --chunk 200 \
    --verbose
