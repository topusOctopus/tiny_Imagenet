import torch.nn as nn


class VGG(nn.Module):
    """
    VGG models

    Args:
         vgg_type (int): Depth of VGG model. Default is 11.
    """
    def __init__(self, vgg_type=11):
        super(VGG, self).__init__()
        vgg_types = [11, 13, 16, 19]
        assert vgg_type in vgg_types, "Wrong value of vgg_type"
        block1 = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                  nn.ReLU()]
        block2 = [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                  nn.ReLU()]
        block3 = [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                  nn.ReLU()]
        block4 = [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                  nn.ReLU()]
        block5 = [nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                  nn.ReLU()]
        block6 = [nn.Linear(512 * 7 * 7, 4096),
                  nn.ReLU(),
                  nn.Dropout(),
                  nn.Linear(4096, 4096),
                  nn.ReLU(),
                  nn.Dropout(),
                  nn.Linear(4096, 200)]

        if vgg_type in vgg_types[1:]:
            block1.extend([nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                           nn.ReLU()])
        if vgg_type in vgg_types[1:]:
            block2.extend([nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                           nn.ReLU()])
        if vgg_type == vgg_types[2]:
            block3.extend([nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                           nn.ReLU()])
        elif vgg_type == vgg_types[3]:
            block3.extend([nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                           nn.ReLU()])
        if vgg_type == vgg_types[2]:
            block4.extend([nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                           nn.ReLU()])
        elif vgg_type == vgg_types[3]:
            block4.extend([nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                           nn.ReLU()])
        if vgg_type == vgg_types[2]:
            block5.extend([nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                           nn.ReLU()])
        elif vgg_type == vgg_types[3]:
            block5.extend([nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                           nn.ReLU()])

        for block in [block1, block2, block3, block4, block5]:
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        block5.append(nn.AdaptiveAvgPool2d((7, 7)))
        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)
        self.block5 = nn.Sequential(*block5)
        self.block6 = nn.Sequential(*block6)
        self._initialize_weights()

    def forward(self, input_data):
        input_data = self.block1(input_data)
        input_data = self.block2(input_data)
        input_data = self.block3(input_data)
        input_data = self.block4(input_data)
        input_data = self.block5(input_data)
        input_data = input_data.view(-1, 512 * 7 * 7)
        input_data = self.block6(input_data)

        return input_data

    def _initialize_weights(self):
        """
        Initialize starting weights of VGG model by Kaiming He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
