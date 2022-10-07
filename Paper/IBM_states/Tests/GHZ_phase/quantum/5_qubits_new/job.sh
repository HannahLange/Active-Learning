#!/bin/bash
#
#SBATCH --job-name=Test
#SBATCH --comment="IBMGHZ5"
#SBATCH --mem=40960
#SBATCH --time=60:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hannah.lange@physik.uni-muenchen.de
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --constraint=avx2


source /project/th-scratch/h/Hannah.Lange/Projektpraktikum/ENVDIR/bin/activate
python3 example.py
