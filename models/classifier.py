import torch.nn as nn
from .cnn_blocks import ConvBlock

class ImageClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_c = cfg['input_channels']
        layers = []
        for out_c in cfg['cnn_blocks']:
            layers.append(ConvBlock(in_c, out_c, cfg['kernel_size'],
                                    cfg['use_batchnorm'],
                                    cfg['use_dropout'],
                                    cfg['dropout_prob']))
            in_c = out_c
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_c * 8 * 8, 128),  # assuming 32x32 input
            nn.ReLU(),
            nn.Linear(128, cfg['num_classes'])
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)
