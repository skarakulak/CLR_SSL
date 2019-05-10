#!/bin/bash
#
# all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=v58_r18
#################
# a file for job output, you can check job progress
#SBATCH --output=output_v58_r18.out
#################
# a file for errors from the job
#SBATCH --error=error_v58_r18.err
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
#SBATCH --cpus-per-task=6
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sk7685@nyu.edu

module load python-3.6
module load cuda-10.0
source /data/sk7685/pytorch_10/bin/activate pytorch_10
#srun python3 ../train15.py -e 130  -v v58_r18 -w wv54_r18 -x wv58_r18 -o sgd -l 0.00001 -d 0.0003 -n 2000 -c .002 -S 0 -W 5 -D 0.2
#srun python3 ../train15.py -e 3000 -v v58_r18 -w wv58_r18 -x wv58_r18 -o sgd -l 0.000003 -d 0.0003 -n 2000 -c .002 -S 0 -W 5 -D 0.2
#srun python3 ../train15.py -e 130  -v v58_r18 -w wv56_r18 -x wv58_r18 -o sgd -l 0.000001  -d 0.0003 -n 2000 -c .002 -S 0 -W 6 -D 0.2 -R 1
#srun python3 ../train15.py -e 3000 -v v58_r18 -w wv58_r18 -x wv58_r18 -o sgd -l 0.0000003 -d 0.0003 -n 2000 -c .002 -S 0 -W 6 -D 0.2 -R 1

srun python3 ../train15.py -e 130  -v v58_r18 -w wv58_r18_best -x wv58_r18 -o sgd -l 0.00003  -d 0.0001 -n 2000 -c .002 -S 0 -W 6 -D 0.2 -R 1
srun python3 ../train15.py -e 3000 -v v58_r18 -w wv58_r18      -x wv58_r18 -o sgd -l 0.00001 -d 0.0001 -n 2000 -c .002 -S 0 -W 6 -D 0.2 -R 1

