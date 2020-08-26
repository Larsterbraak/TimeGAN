#!/bin/bash 
#Set job requirements
#SBATCH -t 5:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu_short
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=larsterbraak@gmail.com

NAME=Master_EONIA.csv # Start environment setup

SCRATCH=/scratch/work/

# Loading modules
module purge #Unload all loaded modules
module load python/3.7.7
module avail tensorflow/2.2.0

#Copy input file to scratch
cp $HOME/$NAME "$TMPDIR"

# Create output directory on scratch
mkdir "$TMPDIR"/output_dir

# Execute a python program 
python $HOME/main.py "$TMPDIR"/$NAME "$TMPDIR"/output_dir

#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME

# Remove the file from scratch - good citizen behaviour
rm -rf "$TMPDIR"/output_dir