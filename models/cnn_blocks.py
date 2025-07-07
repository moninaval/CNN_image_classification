import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, use_bn=True, use_dropout=False, drop_p=0.3):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size=k, padding=k//2),
                  nn.ReLU(inplace=True)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        if use_dropout:
            layers.append(nn.Dropout2d(drop_p))
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
