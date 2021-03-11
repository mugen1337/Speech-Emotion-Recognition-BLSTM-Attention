import sys
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick

import loader
import model
import utils

# features, set
FEAT_PATH  = './Features/LLD'
SET_PATH   = './set/' 
MODEL_PATH = './model/'

# Random Seed Destiny
RANDOM_SEED = 2851

# save name
NAME = 'sample_'  + str(RANDOM_SEED)

LOG = './logs/' + NAME + '.txt'
GRAPH = './logs/' + NAME + '.png'

# model parametor
input_dim     = 32
lstm_hidden   = 32

# other params
batch_size    = 32
learning_rate = 0.0005
max_epoch     = 25
log_interval  = 1

# gpu setting
gpu = 0
use_cuda = torch.cuda.is_available()
if gpu < 0:
    device = "cpu"
else:
    device = torch.device("cuda:"+ str(gpu) if use_cuda else "cpu")
print("device : ", device)

# random seeds
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # slower

setup_seed(RANDOM_SEED)

emotion_label = {
    'ang' : 0,
    'joy' : 1,
    'neu' : 2,
    'sad' : 3,
}

def train(model, criterion, optimizer, train_set):
    model.train()
    print_loss, cnt = 0, 0

    train_set.shuffle()
    data_mask_label = train_set.return_batch()

    while data_mask_label is not None:
        data, mask, label = data_mask_label
        data, mask, label = data.to(device), mask.to(device), label.to(device)
        pred = model(data, mask)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss += loss.data.item() * batch_size
        cnt += data.shape[0]
        data_mask_label = train_set.return_batch()
    print_loss /= cnt
    return print_loss

def test(model, criterion, test_set):
    model.eval()
    print_loss, cnt = 0, 0
    matrix_emotion = [[0 for _ in range(4)] for _ in range(4)]
    test_set.idx = 0
    with torch.no_grad():
        data_mask_label = test_set.return_batch()
        while data_mask_label is not None:
            data, mask, label = data_mask_label
            data, mask, label = data.to(device), mask.to(device), label.to(device)
            pred = model(data, mask)
            loss = criterion(pred, label)
            print_loss += loss.data.item() * batch_size
            cnt += data.shape[0]
            pred = torch.max(pred, 1)[1]
            for fs, sc in zip(label, pred):
                matrix_emotion[int(fs)][int(sc)] += 1
            data_mask_label = test_set.return_batch()
    print_loss /= cnt
    return matrix_emotion, print_loss

def readlist(path):
    with open(path) as f:
        ret = f.read().split(',')
    return ret


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG, mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    train_male   = readlist(SET_PATH + '/train_male.txt')
    train_female = readlist(SET_PATH + '/train_female.txt')
    train_utt    = readlist(SET_PATH + '/train_utt.txt')

    valid_male   = readlist(SET_PATH + '/valid_male.txt')
    valid_female = readlist(SET_PATH + '/valid_female.txt')
    valid_utt    = readlist(SET_PATH + '/valid_utt.txt')

    test_male   = readlist(SET_PATH + '/test_male.txt')
    test_female = readlist(SET_PATH + '/test_female.txt')
    test_utt    = readlist(SET_PATH + '/test_utt.txt')

    train_loader = loader.JTESLoader(
        train_male,
        train_female,
        train_utt,
        utils.emo_li,
        FEAT_PATH,
        input_dim,
        emotion_label,
        40
    )
    mean, std = train_loader.load()

    valid_loader = loader.JTESLoader(
        valid_male,
        valid_female,
        valid_utt,
        utils.emo_li,
        FEAT_PATH,
        input_dim,
        emotion_label,
        40
    )
    valid_loader.load(mean, std)

    test_loader = loader.JTESLoader(
        test_male,
        test_female,
        test_utt,
        utils.emo_li,
        FEAT_PATH,
        input_dim,
        emotion_label,
        40
    )
    test_loader.load(mean, std)

    # model, criterion, optimizer, stopper
    net = model.lld_blstm_attn(hidden_dim=lstm_hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)

    tls, vls, vac = [], [], [] # train loss, valid loss, valid acc

    epoch      = 0
    go_next    = True
    best_epoch = -1
    loss_min   = 1e10

    for epoch in range(1, max_epoch+1):

        # train
        train_loss = train(net, criterion, optimizer, train_loader)

        # valid
        matrix_emotion, valid_loss = test(net, criterion, valid_loader)
        WA, _ = utils.evaluate(matrix_emotion, 4) # JTES is balanced -> UA must be equal to WA

        tls.append(train_loss)
        vls.append(valid_loss)
        vac.append(WA)

        if epoch % log_interval == 0:
            print("\nepoch : {}".format(epoch))
            print("train_loss: \t", train_loss)
            print("valid_loss: \t", valid_loss)
            print(np.array(matrix_emotion))
            print('Acc: {:.6f}'.format(WA))

            logging.info("epoch: {}, train_loss: {:.6f}, valid_loss: {:.6f}, Valid Acc: {:.6f}".format(epoch, train_loss, valid_loss, WA))

        if valid_loss < loss_min:
            torch.save(net.state_dict(), MODEL_PATH + NAME + '.net')
            loss_min = valid_loss
            best_epoch = epoch
            
    

    # train fin
    print("\nbest_epoch : {}".format(best_epoch))
    logging.info("\nbest epoch : {}".format(best_epoch))

    print("\nTest")
    logging.info("\nTest")

    net = model.lld_blstm_attn(hidden_dim=lstm_hidden).to(device)
    net.load_state_dict(torch.load(MODEL_PATH + NAME + '.net'))

    mat_test, test_loss = test(net, criterion, test_loader)
    WA, _ = utils.evaluate(mat_test, 4)
    print(np.array(mat_test))
    print(WA)

    logging.info(np.array(mat_test))
    logging.info(WA)

    utils.make_graph(tls, vls, vac, GRAPH)