#!/bin/bash
ACCOUNT_NAME="spinquest"
PARTITION_NAME="gpu"
CONSUMABLE_RESOURCES="gpu:a100"
CONSTRAINT_NUMBER=16
MAXIMUM_TIMELIMIT="01:00:00"
JOB_NAME="generate_sivers_distributions"
OUTPUT_FILE_FORMAT="txt"
ERROR_FILE_FORMAT="txt"
NUMBER_OF_MEGABYTES_PER_NODE="256000"

#SBATCH --account=${ACCOUNT_NAME}
echo "> Set account as: ${ACCOUNT_NAME}"

#SBATCH --partition=${PARTITION_NAME}
echo "> Set parition as: ${PARTITION_NAME}"

#SBATCH --gres gpu:a100
echo "> Set consumable resources to be: ${CONSUMABLE_RESOURCES}"

#SBATCH --constraint=${CONSTRAINT_NUMBER}
echo "> Set constraint to be: ${CONSTRAINT_NUMBER}"

#SBATCH --time=${MAXIMUM_TIMELIMIT}
echo "> Received ${MAXIMUM_TIMELIMIT} hours as maximum time limit to run the job"

#SBATCH --job-name ${JOB_NAME}
echo "> Received job name: ${JOB_NAME}"

#SBATCH --output result_of_${JOB_NAME}.${OUTPUT_FILE_FORMAT}
echo "> Set error output file and filetype as error_in_${JOB_NAME}.${ERROR_FILE_FORMAT}"

#SBATCH --error error_in_${JOB_NAME}.${ERROR_FILE_FORMAT}
echo "> Set error output file and filetype as error_in_${JOB_NAME}.${ERROR_FILE_FORMAT}"

#SBATCH --mem ${NUMBER_OF_MEGABYTES_PER_NODE}
echo "> Set number of megabytes per node as ${NUMBER_OF_MEGABYTES_PER_NODE}MB/node"

module purge

echo "> module purge successful..."

module load apptainer tensorflow/2.13.0

echo "> loading of apptainer and TensorFlow v2.13.0 successful..."

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif generate_csvs_without_lhapdf.py

echo "> SLURM job finished!"
