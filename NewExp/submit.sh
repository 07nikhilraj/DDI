#!/bin/bash
SCRIPT_NAME=$1
sbatch --export=SCRIPT_NAME="${SCRIPT_NAME}" --job-name="${SCRIPT_NAME}" --output="${SCRIPT_NAME}.%j.out" --error="${SCRIPT_NAME}.%j.err" job.sh
