import random

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick


female_li = [
    "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", 
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", 
    "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", 
    "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39", "f40", 
    "f41", "f42", "f43", "f44", "f45", "f46", "f47", "f48", "f49", "f50"
]

male_li = [
    "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
    "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20", 
    "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30", 
    "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39", "m40", 
    "m41", "m42", "m43", "m44", "m45", "m46", "m47", "m48", "m49", "m50"
]

emo_li = ["ang", "joy", "neu", "sad"]
utt_li = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
        "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
        "41", "42", "43", "44", "45", "46", "47", "48", "49", "50"]


def split_speaker_open_utterance_open_with_test(
    train_male_num=40,
    train_female_num=40,
    train_utt_num=30,
    valid_male_num=5,
    valid_female_num=5,
    valid_utt_num=10,
    test_male_num=5,
    test_female_num=5,
    test_utt_num=10,
    shuffle=False
    ):
    """
    params : 
        nums & shuffle (bool)
    
    ret : 
        train_male, train_female, train_utt,
        valid_male, valid_female, valid_utt,
        test_male,  test_female,  test_utt
    """
    assert train_male_num + valid_male_num + test_male_num == len(male_li), "Error male num is invalid"
    assert train_female_num + valid_female_num + test_female_num == len(female_li), "Error female num is invalid"
    assert train_utt_num + valid_utt_num + test_utt_num == len(utt_li), "Error utt num is invalid"

    if not shuffle:
        ml, fl, ul = 0, 0, 0
        train_male = male_li[ml:ml+train_male_num]
        train_female = female_li[fl:fl+train_female_num]
        train_utt = utt_li[ul:ul+train_utt_num]
        ml += train_male_num
        fl += train_female_num
        ul += train_utt_num
        valid_male = male_li[ml:ml+valid_male_num]
        valid_female = female_li[fl:fl+valid_female_num]
        valid_utt = utt_li[ul:ul+valid_utt_num]
        ml += valid_male_num
        fl += valid_female_num
        ul += valid_utt_num
        test_male = male_li[ml:ml+test_male_num]
        test_female = female_li[fl:fl+test_female_num]
        test_utt = utt_li[ul:ul+test_utt_num]
        return train_male, train_female, train_utt, valid_male, valid_female, valid_utt, test_male, test_female, test_utt

    train_male = list(np.random.choice(male_li, train_male_num, replace=False))
    rem = list(set(male_li)-set(train_male))
    valid_male = list(np.random.choice(rem, valid_male_num, replace=False))
    test_male = list(set(rem)-set(valid_male))

    train_female = list(np.random.choice(female_li, train_female_num, replace=False))
    rem = list(set(female_li)-set(train_female))
    valid_female = list(np.random.choice(rem, valid_female_num, replace=False))
    test_female = list(set(rem)-set(valid_female))

    train_utt = list(np.random.choice(utt_li, train_utt_num, replace=False))
    rem = list(set(utt_li)-set(train_utt))
    valid_utt = list(np.random.choice(rem, valid_utt_num, replace=False))
    test_utt = list(set(rem)-set(valid_utt))
    return train_male, train_female, train_utt, valid_male, valid_female, valid_utt, test_male, test_female, test_utt

def show_heatmap(X, save_to, power_to_DB=False):
    """
    to be fixed -> cant plot 1-dim data
    """
    if power_to_DB:
        X = librosa.power_to_db(X)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(X)
    plt.show()
    plt.savefig(save_to)

def evaluate(matrix, n):
    """
    params:
        matrix (list of list or np.array(n,n) ) : matrix
        n (int) : class num
    return
        WA (float) : Weighted Accuracy
        UA (list of float, len==n) : Unweighted Accuracy
    """
    correct_cnt = 0
    all_cnt = 0
    UA = [0 for _ in range(n)]
    for i in range(n):
        tmp = 0
        correct_cnt += matrix[i][i]
        for j in range(n):
            tmp += matrix[i][j]
        all_cnt += tmp
        if tmp != 0:
            UA[i] = matrix[i][i] / tmp
    WA = correct_cnt / all_cnt

    return WA, UA

def make_graph(train_loss, valid_loss, valid_acc, save_to):
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    valid_acc = np.array(valid_acc)
    x = np.array([i+1 for i in range(len(train_loss))])

    _, ax = plt.subplots(figsize=(5*1.618,5.55))

    ax.plot(x, train_loss, marker='o', label="Train-loss")
    ax.plot(x, valid_loss, marker='x', label="Validation-loss")
    ax.plot(x, valid_acc, marker='+', label="Validation-Accuracy")

    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.tick_params(labelsize=16)
    ax.xaxis.offsetText.set_fontsize(16)
    plt.gca().get_xaxis().set_major_locator(ptick.MaxNLocator(integer=True))
    plt.title("loss/accuracy", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend()
    plt.savefig(save_to)
    plt.close()


def get_wavname(per, emo, utt):
    return '../Data/JTES/jtes_v1.1/wav/' + per + '/' + emo + '/' + per + '_' + emo + '_' + utt + '.wav'
