#!/bin/sh

#SBATCH --account=cbio
#SBATCH --partition=curie
#SBATCH --nodes=1 --ntasks=64
#SBATCH --time=72:00:00
#SBATCH --job-name="RDPSantaSim"
#SBATCH --mail-user=clljos001@myuct.ac.za
#SBATCH --mail-type=ALL

module load java/jdk-11
module load python/miniconda3-py39

#Run python script in the background
python3 Simulation.py -o "/scratch/clljos001" -t 16
