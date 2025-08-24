#!/bin/bash

#SBATCH --job-name=spgemm_matrix    # A descriptive name for your job
#SBATCH --output=slurm-%j.out          # File to write standard output to (%j is the job ID)
#SBATCH --ntasks=1                     # Run only one instance of the program
#SBATCH --cpus-per-task=16             # Request 16 CPU cores for this task
#SBATCH --mem=32G                       # Request 8 Gigabytes of memory
#SBATCH --time=01:00:00                # Set a 30-minute time limit

# --- Job Commands ---

# Set the number of OpenMP threads to match the number of CPU cores requested
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Get the matrix name from the command line argument passed to sbatch
MATRIX_NAME=$1

# Print some useful information to the output file
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Number of OpenMP threads: $OMP_NUM_THREADS"
echo "Benchmarking matrix: $MATRIX_NAME"
echo "-------------------------------------"

# Load the necessary software module on the compute node
module load openmpi/4.1.5

# Run your compiled program
./benchmark $MATRIX_NAME

echo "-------------------------------------"
echo "Job Finished: $(date)"