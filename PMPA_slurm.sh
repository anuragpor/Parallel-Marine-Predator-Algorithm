#!/bin/sh 
#SBATCH -N 16 // specifies number of nodes
#SBATCH --ntasks-per-node=40 //specifies core per node
#SBATCH --time=06:50:20 // specifies maximum duration of run 
#SBATCH --job-name=lammps // specifies job name 
#SBATCH --error=err // specifies error file name 
#SBATCH --output=out //specifies output file name 
#SBATCH --partition=cpu // specifies queue name

### Set environment
module load compiler/intel-mpi/mpi-2018.2.199 
export I_MPI_FABRICS=shm:dapl 
#### Run your command/executable etc
time mpirun -np 5 ./MPA_MPI

