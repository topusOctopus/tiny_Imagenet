import argparse

from torchvision.models import vgg11, resnet18, resnet34, wide_resnet50_2

from models.vgg_model import VGG
from scr.test_time_aug_train.train import train_tt_aug
from scr.simple_aug_train.train import train_simple_aug


def main():
    parser = argparse.ArgumentParser(description="Pytorch image classification")
    parser.add_argument('-b', default=64, type=int, metavar='N',
                        help='mini batch size(default 32) on which we train network each iter')
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-iter', default=100, type=int, metavar='N',
                        help='initial iteration number after print training results')
    parser.add_argument('--topk', default=(1,), type=tuple, metavar='T',
                        help='determine top k result of trained nn')
    parser.add_argument('--dataset_dir', default='../data/tiny-imagenet-200/',
                        type=str, metavar='PATH', help='path to training data')
    parser.add_argument('--epoch_to_save', default=1, type=int, metavar='N',
                        help='epochs num after save checkpoint')
    parser.add_argument('-sp', '--save_path',
                        default='../models/checkpoints/',
                        type=str, metavar='PATH', help='path to save checkpoint of the model')
    parser.add_argument('--lr_step', default=15, type=int,
                        help="After this number of epochs drop learning rate")
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help="Number to decrease learning rate")
    parser.add_argument('-l2', default=0.0001, type=float,
                        help="Weight decay")
    args = parser.parse_args()

    # train_tt_aug(args, VGG())
    train_simple_aug(args, resnet18())

if __name__ == "__main__":
    main()
