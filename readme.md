# Marine Predators Algorithm (MPA) with MPI

This project involves the implementation and optimization of the Marine Predators Algorithm (MPA) in a parallel computing environment using the Message Passing Interface (MPI) library.

## Overview

The Marine Predators Algorithm is a metaheuristic optimization algorithm inspired by the hunting behavior of marine predators such as sharks.

In this project, the MPA algorithm has been parallelized using MPI, which allows the algorithm to run on multiple processors simultaneously. Each processor maintains a sub-population of solutions (or "predators"), and these sub-populations evolve independently in parallel.

## Features

- **Parallelization**: The MPA algorithm has been parallelized using MPI, allowing it to handle larger problem sizes and run faster on multi-processor systems.
- **Population Shuffling with MPI_Gather and MPI_Scatter**: A shuffling operation is performed every 10th iteration, which uses `MPI_Gather` to collect all solutions from all processors to the root processor. The root processor then shuffles these solutions and uses `MPI_Scatter` to distribute them back to all processors. This operation increases the diversity of the population and prevents the algorithm from getting stuck in local optima, thereby enhancing the exploration and exploitation capabilities of the algorithm.

## Compilation and Setup

Before compiling the program, make sure to install or load the OpenMPI library in your multiprocessing environment or supercomputer.

To compile the program, use the following command:

```bash
mpic++ -std=c++11 -o mpa MPA.cpp
```
To run the program on multiple processors, use the following command:
```bash
mpirun -np num_processors ./MPA_MPI
```
Replace num_processors with the number of processors you want to use.