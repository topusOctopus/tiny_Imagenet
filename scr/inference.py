from __future__ import print_function, division, absolute_import
import argparse
import torch
import os
import sys
import torch.nn as nn
import pretrainedmodels.utils as utils
from PIL import Image
from torchvision.models import wide_resnet50_2

sys.path.append('.')


def run_eval(img_path, model):
    model = model
    model.eval()

    # Load and Transform one input image
    tf_img = utils.TransformImage(model)
    img = img_path
    input_data = Image.open(img)  # 3x400x225
    input_data = tf_img(input_data)  # 3x299x299
    input_data = input_data.unsqueeze(0)  # 1x3x299x299
    input = torch.autograd.Variable()

    # Load Imagenet Synsets
    with open('../data/tiny-imagenet-200/words.txt', 'r') as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(' ') for line in synsets]
    key_to_classname = {spl[0]: ' '.join(spl[1:]) for spl in splits}

    with open('../data/tiny-imagenet-200/wnids.txt', 'r') as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]

    # Make predictions
    output = model(input)  # size(1, 1000)
    max_, argmax = output.data.squeeze().max(0)
    class_id = argmax.item()
    class_key = class_id_to_key[class_id]
    classname = key_to_classname[class_key]
    return img, classname, max_, class_key


def main():
    arch = 'wide_resnet50_2'
    parser = argparse.ArgumentParser(description='Resnet50')
    parser.add_argument('--img_path', type=str, default='/home/topus/Desktop/tiny_imagenet/data/tiny-imagenet-200/test/images/test_1.JPEG', metavar='PATH',
                        help='path to get image from repository')
    args = parser.parse_args()
    img_path = args.img_path
    #model = resnet50(num_classes=1000, pretrained='imagenet')
    model = wide_resnet50_2(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 200)
    #model.load_state_dict(torch.load(os.path.join("../models/load_path", "model_ep31.pth"), map_location=torch.device('cpu')))
    checkpoint = torch.load(os.path.join("../models/load_path", "model_ep31.pth"),  map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img, classname, max_, class_key = run_eval(img_path, model)
    print("'{}': '{}' is a '{}' with {}% confidence".format(arch, img, classname, round(max_.item() * 100, 2)))


if __name__ == '__main__':
    main()
