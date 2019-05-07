#!/bin/bash
#
# all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=V30_r18
#################
# a file for job output, you can check job progress
#SBATCH --output=output_v30_r18.out
#################
# a file for errors from the job
#SBATCH --error=error_v30_r18.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the # faster your job will run.
# Default is one hour, this example will run in  less that 5 minutes.
#SBATCH --time=1-23:58:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
#SBATCH --gres gpu:1
#SBATCH --constraint=gpu_12gb
# We are submitting to the batch partition
#SBATCH --qos=batch
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=12000
#SBATCH --cpus-per-task=4
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sk7685@nyu.edu

module load python-3.6
module load cuda-10.0
source /data/sk7685/pytorch_10/bin/activate pytorch_10
srun python3 ../train11.py -e 300  -v v30_r18 -w wv30_r18 -x wv30_r18 -o adam -l 0.001  -N 6000 -f 0 -k 1
srun python3 ../train11.py -e 3000 -v v30_r18 -w wv30_r18 -x wv30_r18 -o adam -l 0.0003 -N 6000 -f 0 -k 1


