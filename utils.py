import os
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

torch.manual_seed(0)
np.random.seed(0)


def prepare_datasets(speech_path='./data/speech_train.npy'
                     , music_path='./data/music_train.npy'
                     , splits=[0.7, 0.15, 0.15]):
    ######################################################
    ################### Preparing Data ###################
    # speech_path: 	Path to speech npy file
    # music_path: 	Path to music npy file
    # splits: 		list of split percentages for dataset
    ######################################################

    assert np.sum(splits) == 1
    assert splits[0] != 0
    assert splits[1] != 0
    assert splits[2] != 0

    ########### load data into torch Tensors #############

    speech_train = torch.Tensor(np.load(speech_path))
    music_train = torch.Tensor(np.load(music_path))

    ###### generate labels: Speech = 0; Music= 1 #########

    labels_speech = torch.LongTensor(np.zeros(speech_train.size(0)))
    labels_music = torch.LongTensor(np.ones(music_train.size(0)))

    X = torch.cat((speech_train, music_train))
    y = torch.cat((labels_speech, labels_music))

    ###### split dataset into training validation ########
    ###### and test. 0.7, 0.15, 0.15 split        ########

    n_points = y.size(0)

    train_split = (0, int(splits[0] * n_points))
    val_split = (train_split[1], train_split[1] + int(splits[1] * n_points))
    test_split = (val_split[1], val_split[1] + int(splits[2] * n_points))

    shuffle_indices = np.random.permutation(np.arange(n_points))

    train_indices = torch.LongTensor(shuffle_indices[train_split[0]:train_split[1]])
    val_indices = torch.LongTensor(shuffle_indices[val_split[0]:val_split[1]])
    test_indices = torch.LongTensor(shuffle_indices[test_split[0]:test_split[1]])

    train_set = (X[train_indices], y[train_indices])
    val_set = (X[val_indices], y[val_indices])
    test_set = (X[test_indices], y[test_indices])

    ############# create torch Datasets ##################

    train_set = torch.utils.data.TensorDataset(train_set[0], train_set[1])
    val_set = torch.utils.data.TensorDataset(val_set[0], val_set[1])
    test_set = torch.utils.data.TensorDataset(test_set[0], test_set[1])

    return train_set, val_set, test_set


def evaluate(data_loader, model, criterion, cuda):
    ######################################################
    ################### Evaluate Model ###################
    # data_loader: 	pytorch dataloader for eval data
    # model: 		pytorch model to be evaluated
    # criterion: 	loss function used to compute loss
    # cuda:			boolean for whether to use gpu

    # Returns loss and accuracy
    ######################################################
    ######## WRITE YOUR CODE BELOW #######################
    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        # Keep track of how the model dows
        loss = 0
        correct = 0
        n_examples = 0

        for batch_i, batch in enumerate(data_loader):
            data, target = batch
            if cuda:
                data, target = data.cuda(), target.cuda()

            # Create variables
            data, target = Variable(data), Variable(target)

            # Make prediction
            output = model(data)

            # Get argmax of probabilities
            pred = output.data.max(1, keepdim=True)[1]

            # Add number of correct predictions
            correct += pred.eq(target.data.view_as(pred)).sum()

            # Add loss
            loss += criterion(output, target).data[0]

            # Add total number of predictions made
            n_examples += pred.size(0)

        # Average loss
        loss /= n_examples

        # Accuracy percentage
        accuracy = correct.item() / n_examples

    return loss, accuracy
######################################################


def save(model, path):
    ######################################################
    ################### Save Model ###################
    # model: 	pytorch model to be saved
    # path:		path for model to be saved
    ######################################################
    ######## WRITE YOUR CODE BELOW #######################
    torch.save(model, "models/" + path + '.pth')


def load(path):
    ######################################################
    ################### Load Model ###################
    # path:		path of model to be loaded

    # Returns model state_dict
    ######################################################
    ######## WRITE YOUR CODE BELOW #######################
    state_dict = torch.load("models/" + path + '.pth')
    return state_dict
