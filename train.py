import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from utils import prepare_datasets, evaluate, save, load
from models import ConvNet, MyModel
# Training settings
parser = argparse.ArgumentParser(description='HW 2: Music/Speech CNN')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate')
# parser.add_argument('--momentum', type=float, metavar='M',
#                     help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', default=50,
                    help='number of epochs to train')
parser.add_argument('--model', default='convnet',
                    choices=['convnet', 'mymodel'],
                    help='which model to train/evaluate')
parser.add_argument('--save-dir', default='models/')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(0)
np.random.seed(0)
if args.cuda:
    torch.cuda.manual_seed(0)


############# fetch torch Datasets ###################
######### you may change the dataset split % #########
train_set, val_set, test_set = prepare_datasets(splits = [0.7, 0.15, 0.15])


############# create torch DataLoaders ###############
########### you may change the batch size ############
train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = 1000)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1000)


################ initialize the model ################
if args.model == 'convnet':
    model = ConvNet()
elif args.model == 'mymodel':
    model = MyModel()
else:
    raise Exception('Incorrect model name')

if args.cuda:
    model.cuda()

######## Define loss function and optimizer ##########
############## Write your code here ##################
params = model.parameters()
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
######################################################


def train(epoch):
    """ Runs training for 1 epoch
    epoch: int, denotes the epoch number for printing
    """
    ############# Write train function ###############
    mean_training_loss = 0.0
    model.train()
    for i, batch in enumerate(train_loader):
        ############ Write your code here ############
        # Get input and labels
        spectograms, targets = Variable(batch[0]), Variable(batch[1])

        # Zero out gradients
        optimizer.zero_grad()

        # Do forward pass
        output = model(spectograms)

        # Calculate loss
        loss = criterion(output, targets)
        # Add to mean
        mean_training_loss += loss

        # Backprop
        loss.backward()

        # Optimize
        optimizer.step()

    mean_training_loss = mean_training_loss / len(train_loader)
    print('Training Epoch: [{}]\t'
            'Training Loss: {:.6f}'.format(
            (epoch), mean_training_loss))
    ##################################################

######## Training and evaluation loop ################
######## Save model with best val accuracy  ##########
most_acc_model = None
highest_val_acc = -1
for i in range(args.epochs):
    train(i)
    val_loss, val_acc = evaluate(val_loader, model, criterion, args.cuda)
    print('Validation Loss: {:.6f} \t'
            'Validation Acc.: {:.6f}'.format(
            val_loss, val_acc))
    ####### write saving code here ###################
    if val_acc > highest_val_acc:
        highest_val_acc = val_acc
        most_acc_model = model

save(model, args.model)

############ write testing code here #################
def test(model):
    test_loss, test_acc = evaluate(val_loader, model, criterion, args.cuda)
    print('Test Loss: {:.6f} \t'
            'Test Acc.: {:.6f}'.format(
            test_loss, test_acc))


############# Load best model and test ###############
############## Write your code here ##################
model = load(args.model)
test(model)