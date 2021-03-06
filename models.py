import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #############################################################################
        # TODO: Implement the ConvNet as per the instructions given in the assignment
        # 3 layer CNN followed by 1 fully connected layer. Use only ReLU activation
        #   CNN #1: k=8, f=3, p=1, s=1
        #   Max Pool #1: pooling size=2, s=2
        #   CNN #2: k=16, f=3, p=1, s=1
        #   Max Pool #2: pooling size=4, s=4
        #   CNN #3: k=32, f=3, p=1, s=1
        #   Max Pool #3: pooling size=4, s=4
        #   FC #1: 64 hidden units                                                             
        ############################################################################# 
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.fc = nn.Linear(512, 64)
        self.classifier = nn.Linear(64, 2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass for the model
        #############################################################################
        out = self.layer_1(torch.unsqueeze(x, 1))
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.classifier(out)

        return out
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Implement your own model based on the hyperparameters of your choice
        #############################################################################
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.fcn = nn.Conv3d(1, 256, (32, 4, 4), padding=0, stride=1)
        self.linear = nn.Sequential(
            nn.Linear(256, 127),
            nn.ReLU(),
            nn.Linear(127, 2)
        )

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass for your model
        ############################################################################# 
        out = self.layer_1(torch.unsqueeze(x, 1))
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.fcn(torch.unsqueeze(out, 1))
        out = torch.squeeze(out)
        out = self.linear(out)

        return out
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################