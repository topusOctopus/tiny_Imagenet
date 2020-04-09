import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk):
    """
    Computes accuracy over the k top predictions for the specified values of k
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # conf_matrix = confusion_matrix(target.view(-1), pred)
        # print(conf_matrix)
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# not using currently
def C_matrix(actual, predicted, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Compute and plot confusion matrix
    """

    if not title:
        if not normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion maxtrix without normalization'

    # compute c_matrix
    cm = confusion_matrix(actual, predicted)
    # classes = classes[unique_labels(actual, predicted)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix:\n', cm)
    else:
        print('Confusion maxtrix without normalization:\n', cm)

    fig, ax = plt.subplots()
    img = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(img, ax=ax)
    # show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
