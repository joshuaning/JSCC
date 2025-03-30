# MINE = Mutual Information Neural Estimation
# from https://arxiv.org/abs/1801.04062 (2018)
# code is a rewrote version of https://github.com/13274086/DeepSC/tree/master

import torch
import torch.nn as nn
import torch.nn.functional as F

# DeepSC's MINE FC weight initialization
def linear_custom_weights(in_dim, out_dim):
    l = nn.Linear(in_dim, out_dim)
    l.weight = torch.nn.Parameter(torch.normal(0, 0.2, size=l.weight.shape))
    l.bias.data.zero_()
    return l

# 3 FC layers to estimate the mutual information of the inputs and outputs of the channel (AWGN)
# input_dim = 2 because IQ data
# TODO:: Experiment with a channel estimator inconjunction or replacing the MINE
class MINE(nn.Module):
    def __init__(self, hidden_dim=10, input_dim=2, default_weights=False):
        super().__init__()
        
        '''
        DeepSC used a custom weight initialization.
        The paper explained the effect of learning rate on model convergence,
        but did not comment on the weight initialization.
        We are using the same parameters by default to recreate performance,
        but we will also experiment with the default initialization with a 
        bool toggle called default_weights 
        '''
        if default_weights:
            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, 1)
        else:
            self.l1 = linear_custom_weights(input_dim, hidden_dim)
            self.l2 = linear_custom_weights(hidden_dim, hidden_dim)
            self.l3 = linear_custom_weights(hidden_dim, 1)

    def forward(self, X_in_data):

        x = self.l1(X_in_data)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        Y_out_data = self.l3(x)

        return Y_out_data

# calculate variables used to calculate moving_average_e_T and MI_lower_bound
# following equations from the MINE paper
def mutual_information(joint, marginal, MINE_net):
    T = MINE_net(joint)
    e_T = torch.exp(MINE_net(marginal))
    MI_lower_bound = torch.mean(T) - torch.log(torch.mean(e_T))

    return MI_lower_bound, T, e_T

# returns loss, moving_average_e_T, MI_lower_bound
def learn_MINE(batch, MINE_net, moving_average_e_T, moving_average_rate=0.01):

    #batch is a tuple of (joing, marginal)
    joint, marginal = batch
    joint = torch.FloatTensor(joint)
    marginal = torch.FloatTensor(marginal)

    MI_lower_bound, T, e_T = mutual_information(joint, marginal, MINE_net)
    moving_average_e_T = (1 - moving_average_rate) * moving_average_e_T + moving_average_rate * torch.mean(e_T)
    
    #unbiased with moving average
    loss = -(torch.mean(T) - (1 / torch.mean(moving_average_e_T)) * torch.mean(e_T))

    return loss, moving_average_e_T, MI_lower_bound


# returns the joint and marginal based on the Transmitted and Recived data
# Tx and Rx is complex and they represent the IQ
def sample_batch(Tx, Rx):
    Tx = torch.reshape(Tx, shape=(-1,1))
    Rx = torch.reshape(Rx, shape=(-1,1))

    Tx_samp1, Tx_samp2 = torch.split(Tx, int(Tx.shape[0] / 2), dim=0)
    Rx_samp1, Rx_samp2 = torch.split(Rx, int(Rx.shape[0] / 2), dim=0)

    #construct joint and marginal
    joint = torch.cat((Tx_samp1, Rx_samp1), 1)
    marginal = torch.cat((Tx_samp1, Rx_samp2), 1)

    return joint, marginal
