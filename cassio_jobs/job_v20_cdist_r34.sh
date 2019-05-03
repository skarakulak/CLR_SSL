#!/bin/bash
#
# all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=V20_r34
#################
# a file for job output, you can check job progress
#SBATCH --output=output_v20_r34.out
#################
# a file for errors from the job
#SBATCH --error=error_v20_r34.err
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
#srun python3 ../train9.py -s 0 -e 60   -a resnet34 -v v20_r34 -w wv17_r34 -x wv20_r34 -o sgd -l 0.001  -T 0.2 -d 0.3 -c 15. -p 2000 -n 4 -N 3000 -C cassio -u sk7685
#srun python3 ../train9.py -s 0 -e 3000 -a resnet34 -v v20_r34 -w wv20_r34 -x wv20_r34 -o sgd -l 0.0003 -T 0.2 -d 0.3 -c 15. -p 2000 -n 4 -N 3000 -C cassio -u sk7685
srun python3 ../train9.py -s 0 -e 60   -a resnet34 -v v20_r34 -w wv17_r34 -x wv20_r34 -o sgd -l 0.00003  -T 0.2 -d 0.3 -c 15. -p 2000 -n 4 -N 3000 -C cassio -u sk7685
srun python3 ../train9.py -s 0 -e 3000 -a resnet34 -v v20_r34 -w wv20_r34 -x wv20_r34 -o sgd -l 0.00001 -T 0.2 -d 0.3 -c 15. -p 2000 -n 4 -N 3000 -C cassio -u sk7685

