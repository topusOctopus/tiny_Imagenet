import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as f
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from logger import logger
from torchvision.models import resnet18, vgg11

from models.vgg_model import VGG
from utils.accuracy_metric import accuracy, AverageMeter


def val(criterion, running_loss, epoch, args, val_loader, model, device):
    """
    Validate model on validation set

    Args:
        criterion: CrossEntropyLoss function
        running_loss: (float). Loss value
        epoch: (int). Number of iterations model will train
        args: Argument parser
        val_loader: Validation set DataLoader
        model: Training model
        device: Use GPU or CPU to make calculations

    Return:
        top1: Average validation accuracy
        losses: Average validation loss
    """

    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    with torch.no_grad():
        for i, data_ in enumerate(val_loader):
            inputs, labels = data_
            inputs = inputs.to(device)
            labels = labels.to(device)
            bs, ncrops, c, h, w = inputs.size()
            inputs = model(inputs.view(-1, c, h, w))
            outputs = inputs.view(bs, ncrops, -1).mean(1)
            loss = criterion(outputs, labels)

            topks = accuracy(outputs, labels, topk=args.topk)
            losses.update(loss.item(), inputs.size(0))
            top1.update(topks[0], inputs.size(0))
            # print results
            running_loss += loss.item()
            running_loss = 0.0

    print('Epoch-{} avg_loss_val: {}'.format(epoch, losses))
    print('Epoch-{} avg_acc_val: {}'.format(epoch, top1))
    return top1, losses


def train_tt_aug(args, model):
    """
    Loading data and training ANN model

    Args:
        args: Argument parser
        model: ANN model to train

    Return:
        train_loader: Train set DataLoader
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_type = str(type(model).__name__)
    model_types = [VGG(), resnet18(), vgg11()]
    model_types = [str(type(obj).__name__) for obj in model_types]
    if model_type == model_types[0]:
        pass
    elif model_type == model_types[1]:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 200)
    elif model_type == model_types[2]:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 200)
    else:
        assert model in model_types, "Wrong value for model object"
    if model_type in model_types[1:]:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    model = model.to(device)

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # train augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.TenCrop(56),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack([f.normalize(crop) for crop in crops]))
        ])

    # validation augmentation
    transform_val = transforms.Compose(
        [
            transforms.TenCrop(56),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack([f.normalize(crop) for crop in crops]))
        ])

    # train dataset and data_loader
    train_dir = os.path.join(args.dataset_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = data.DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    logger.info("Loaded: %s", train_loader)

    # val dataset and data_loader
    val_dir = os.path.join(args.dataset_dir, 'new_val')
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
    val_loader = data.DataLoader(val_dataset, batch_size=args.b, shuffle=True)
    logger.info("Loaded: %s", val_loader)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # training
    for epoch in range(args.epochs):
        model.train()
        logger.info("-- Epoch: %s", epoch)
        running_loss = 0.0
        for i, data_ in enumerate(train_loader):
            inputs, labels = data_
            inputs = inputs.to(device)
            labels = labels.to(device)
            bs, ncrops, c, h, w = inputs.size()
            inputs = model(inputs.view(-1, c, h, w))
            outputs = inputs.view(bs, ncrops, -1).mean(1)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            topks = accuracy(outputs, labels, topk=args.topk)
            top1.update(topks[0], inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print results on train
            running_loss += loss.item()
            if i % args.iter == (args.iter - 1):
                print('[%d, %5d] loss_train: %.3f' %
                      (epoch + 1, i + 1, running_loss / args.iter))
                for topk_accuracy, k in zip(topks, args.topk):
                    print("top-{} accuracy_train: {}".format(k, topk_accuracy))
                running_loss = 0.0

        print('\n')
        logger.info('Epoch-{} avg_loss_train: {}'.format(epoch, losses))
        logger.info('Epoch-{} avg_acc_train: {}'.format(epoch, top1))
        print('\n')

        # checking on validation set
        val(criterion, running_loss, epoch, args, val_loader, model, device)

        # saving model
        if epoch % args.epoch_to_save == 0:
            model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'accuracy': topks[0]
            }
            torch.save(model_checkpoint, os.path.join(args.save_path, 'model_ep{}.pth'.format(epoch)))
        scheduler.step()
    logger.info('Finished Training')
    return train_loader

