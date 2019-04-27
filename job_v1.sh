#!/bin/bash
#
# all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=CITY_V20
#################
# a file for job output, you can check job progress
#SBATCH --output=output_v20.out
#################
# a file for errors from the job
#SBATCH --error=error_v20.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the # faster your job will run.
# Default is one hour, this example will run in  less that 5 minutes.
#SBATCH --time=1-23:58:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
#SBATCH --gres gpu:p40:1
#remove SBATCH --constraint=gpu_12gb
# We are submitting to the batch partition
# remove SBATCH --qos=batch
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=12000
#SBATCH --cpus-per-task=3
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sk7685@nyu.edu

module load python3/intel/3.6.3
module load cuda/10.0.130
#virtualenv ~/pytorch_10
source ~/pytorch_10/bin/activate pytorch_10
srun python3 train.py -s 0 -e 25 -a resnet18 -v v1 -w wv1 -x wv1 -o adam -l 0.01 -d 0.3 -m 1 -c 1 -p 500
