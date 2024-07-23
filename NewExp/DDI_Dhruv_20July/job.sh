#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
##SBATCH -n 64
#SBATCH -c 64
#SBATCH --mem 200G
#SBATCH -t 3-20:00
#SBATCH --mail-user=f20200093@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL
spack load anaconda3/tzdgetu
source activate cardio
python3 "${SCRIPT_NAME}.py"
