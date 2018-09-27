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
        pass
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass for the model
        ############################################################################# 
        return None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Implement your own model based on the hyperparameters of your choice
        ############################################################################# 
        pass
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass for your model
        ############################################################################# 
        return None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################