import argparse
from model3 import *


parser = argparse.ArgumentParser(description='Trains semi-supervised model on the given dataset')
parser.add_argument('-s','--start-epoch', type=int, default=0, help='index of the first epoch')
parser.add_argument('-e','--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-a','--arch', type=str, default='resnet18', help='model architecture (resnet18/resnet34')
parser.add_argument('-v','--version', type=str, default='v0', help='version of the model')
parser.add_argument('-w','--weights-version-load', type=str, default='v0', help='version of the model weight checkpoint to load')
parser.add_argument('-x','--weights-version-save', type=str, default='v0', help='version of the model weight checkpoint to load')
parser.add_argument('-o','--set-optimizer', type=str, default='adam', help="optimizer ('adam' or 'sgd')")
parser.add_argument('-l','--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-d','--dropout', type=float, default=0.2, help='dropout probability for uncertainty loss')
parser.add_argument('-m','--coef-uncertainty-loss', type=float, default=1, help='multiplier for the  uncertainty loss')
parser.add_argument('-c','--coef-unsup-ent-loss', type=float, default=1, help='percentage of unsup entropy in the total loss')
parser.add_argument('-p','--print-freq', type=int, default=100, help='multiplier for the variance loss')
parser.add_argument('-n','--num-of-workers', type=int, default=3, help='number of workers for the dataprep')
parser.add_argument('-C','--hpc', type=str, default='prince', help='prince/cassio/bigpurple')
parser.add_argument('-u','--user', type=str, default='sk7685', help='hpc username to be used when determining paths')

args = parser.parse_args()
print('\nversion name: ' + args.version +'\n')

train_and_val(args)


