"""
Implementation of original GoogLenet architecture in PyTorch, from the paper
"Going Deeper with Convolutions" by Christian Szegedy et al.
at: https://arxiv.org/pdf/1409.4842v1.pdf
"""


# importing necessary libraries

import torch
import torch.nn as nn




# InceptionNet model based on Inception V1 module
# also known as the GoogLeNet

class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_1 = conv_block(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.conv_2 = conv_block(in_channels = 64, out_channels = 192, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.inception_3a = inception_module(in_channels = 192, out_1x1 = 64, in_3x3 = 96, out_3x3 = 128, in_5x5 = 16, out_5x5 = 32, out_maxpool = 32)
        self.inception_3b = inception_module(in_channels = 256, out_1x1 = 128, in_3x3 = 128, out_3x3 = 192, in_5x5 = 32, out_5x5 = 96, out_maxpool = 64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.inception_4a = inception_module(in_channels = 480, out_1x1 = 192, in_3x3 = 96, out_3x3 = 208, in_5x5 = 16, out_5x5 = 48, out_maxpool = 64)
        self.inception_4b = inception_module(in_channels = 512, out_1x1 = 160, in_3x3 = 112, out_3x3 = 224, in_5x5 = 24, out_5x5 = 64, out_maxpool = 64)
        self.inception_4c = inception_module(in_channels = 512, out_1x1 = 128, in_3x3 = 128, out_3x3 = 256, in_5x5 = 24, out_5x5 = 64, out_maxpool = 64)
        self.inception_4d = inception_module(in_channels = 512, out_1x1 = 112, in_3x3 = 144, out_3x3 = 288, in_5x5 = 32, out_5x5 = 64, out_maxpool = 64)
        self.inception_4e = inception_module(in_channels = 528, out_1x1 = 256, in_3x3 = 160, out_3x3 = 320, in_5x5 = 32, out_5x5 = 128, out_maxpool = 128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.inception_5a = inception_module(in_channels = 832, out_1x1 = 256, in_3x3 = 160, out_3x3 = 320, in_5x5 = 32, out_5x5 = 128, out_maxpool = 128)
        self.inception_5b = inception_module(in_channels = 832, out_1x1 = 384, in_3x3 = 192, out_3x3 = 384, in_5x5 = 48, out_5x5 = 128, out_maxpool = 128)
        self.avgpool_5 = nn.AvgPool2d(kernel_size = 7, stride = 1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p = 0.4)
        self.fc = nn.Linear(in_features = 1024, out_features = num_classes)



        self.aux_1 = auxiliary_classifier(in_channels = 512, num_classes = num_classes)
        self.aux_2 = auxiliary_classifier(in_channels = 528, num_classes = num_classes)


    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.maxpool_2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool_3(x)

        x = self.inception_4a(x)

        if self.training:
            aux1 = self.aux_1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)

        if self.training:
            aux2 = self.aux_2(x)

        x = self.inception_4e(x)
        x = self.maxpool_4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool_5(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training:
            return aux1, aux2, x
        else:
            return x


# inception module

class inception_module(nn.Module):
    def __init__(self, in_channels, out_1x1, in_3x3, out_3x3, in_5x5, out_5x5, out_maxpool):
        super().__init__()

        self.branch_1 = conv_block(in_channels = in_channels, out_channels = out_1x1, kernel_size = 1, stride = 1, padding = 0)

        self.branch_2 = nn.Sequential(
            conv_block(in_channels = in_channels, out_channels = in_3x3, kernel_size = 1, stride = 1, padding = 0),
            conv_block(in_channels = in_3x3, out_channels = out_3x3, kernel_size = 3, stride = 1, padding = 1)
        )

        self.branch_3 = nn.Sequential(
            conv_block(in_channels = in_channels, out_channels = in_5x5, kernel_size = 1, stride = 1, padding = 0),
            conv_block(in_channels = in_5x5, out_channels = out_5x5, kernel_size = 5, stride = 1, padding = 2)
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            conv_block(in_channels = in_channels, out_channels = out_maxpool, kernel_size = 1, stride = 1, padding = 0)
        )


    def forward(self, x):
        return torch.cat(
            [self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], 1
        )


# auxiliary classifiers

class auxiliary_classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool = nn.AvgPool2d(kernel_size = 5, stride = 3)
        self.conv_1 = conv_block(in_channels = in_channels, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.fc1 = nn.Linear(in_features = 4 * 4 * 128, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv_1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)



# convolution block

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))  