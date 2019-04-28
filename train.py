import argparse
from model import *


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
parser.add_argument('-c','--coef-unsup-ent-loss', type=float, default=1, help='multiplier for the entropy loss')
parser.add_argument('-p','--print-freq', type=int, default=100, help='multiplier for the variance loss')

args = parser.parse_args()
print('\nversion name: ' + args.version +'\n')

train_and_val(args)


