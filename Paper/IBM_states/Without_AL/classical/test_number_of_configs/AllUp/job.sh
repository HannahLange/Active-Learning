#!/bin/bash
#
#SBATCH --job-name=LGTModel
#SBATCH --comment="All_TestIBM"
#SBATCH --mem=40960
#SBATCH --time=40:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hannah.lange@physik.uni-muenchen.de
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --partition=th-cl,th-ws
#SBATCH --constraint=avx2


source /project/th-scratch/m/Maximilian.Buser/syten_pub/sytensetup
source /project/th-scratch/h/Hannah.Lange/Projektpraktikum/ENVDIR/bin/activate
python3 qiskit_example.py
