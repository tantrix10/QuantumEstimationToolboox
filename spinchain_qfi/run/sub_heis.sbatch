#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=3882
#SBATCH --time=48:00:00




module purge
module load intel/2018.3.222-GCC-7.3.0-2.30  impi/2018.3.222
module load Python/3.6.6



python ./run.py
