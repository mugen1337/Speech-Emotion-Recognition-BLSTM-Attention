import os
import sys
import re
import glob
import shutil
import csv

import numpy as np

sys.path.append('..')
from utils import *


# REWRITE HERE
wav_dir  = '../../Data/JTES/jtes_v1.1/wav/'
opensmile_dir = "../../opensmile-2.3.0"


conf_file = './smile/lld_seq.conf'
save_dir = './LLD/'


per_li = [
    "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", 
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", 
    "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", 
    "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39", "f40", 
    "f41", "f42", "f43", "f44", "f45", "f46", "f47", "f48", "f49", "f50",
    "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
    "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20", 
    "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30", 
    "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39", "m40", 
    "m41", "m42", "m43", "m44", "m45", "m46", "m47", "m48", "m49", "m50"
]
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
utt_li = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50"
]

index = {'name':0, 'frameTime':1, 'pcm_RMSenergy_sma':2, 'pcm_fftMag_mfcc_sma[1]':3, 'pcm_fftMag_mfcc_sma[2]':4, 'pcm_fftMag_mfcc_sma[3]':5, 'pcm_fftMag_mfcc_sma[4]':6, 'pcm_fftMag_mfcc_sma[5]':7, 'pcm_fftMag_mfcc_sma[6]':8, 'pcm_fftMag_mfcc_sma[7]':9, 'pcm_fftMag_mfcc_sma[8]':10, 'pcm_fftMag_mfcc_sma[9]':11, 'pcm_fftMag_mfcc_sma[10]':12, 'pcm_fftMag_mfcc_sma[11]':13, 'pcm_fftMag_mfcc_sma[12]':14, 'pcm_zcr_sma':15, 'voiceProb_sma':16, 'F0_sma':17, 'pcm_RMSenergy_sma_de':18, 'pcm_fftMag_mfcc_sma_de[1]':19, 'pcm_fftMag_mfcc_sma_de[2]':20, 'pcm_fftMag_mfcc_sma_de[3]':21, 'pcm_fftMag_mfcc_sma_de[4]':22, 'pcm_fftMag_mfcc_sma_de[5]':23, 'pcm_fftMag_mfcc_sma_de[6]':24, 'pcm_fftMag_mfcc_sma_de[7]':25, 'pcm_fftMag_mfcc_sma_de[8]':26, 'pcm_fftMag_mfcc_sma_de[9]':27, 'pcm_fftMag_mfcc_sma_de[10]':28, 'pcm_fftMag_mfcc_sma_de[11]':29, 'pcm_fftMag_mfcc_sma_de[12]':30, 'pcm_zcr_sma_de':31, 'voiceProb_sma_de':32, 'F0_sma_de':33}

def make_meta_dic(wav_dir):
    wav_files = glob.glob(wav_dir + '/*.wav')

    print("Loading files ... ")

    meta_dic = {}
    cnt = 0

    # male
    for p in male_li:
        for e in emo_li:
            for u in utt_li:
                gender = "M"
                per_id = p[1:3]
                utt_id = u

                meta_dic[cnt] = {
                    "gender"  : gender,
                    "emotion" : e,
                    "per_id"  : per_id,
                    "utt_id"  : utt_id,
                    "path"    : get_wavname(p, e, u)
                }
                cnt += 1
            
    # female
    for p in female_li:
        for e in emo_li:
            for u in utt_li:
                gender = "F"
                per_id = p[1:3]
                utt_id = u

                meta_dic[cnt] = {
                    "gender"  : gender,
                    "emotion" : e,
                    "per_id"  : per_id,
                    "utt_id"  : utt_id,
                    "path"    : get_wavname(p, e, u)
                }
                cnt += 1
    print("loaded ", len(meta_dic), " files")
    return meta_dic

if __name__ == '__main__':

    meta_dic = make_meta_dic(wav_dir)

    print("Start !")

    for i in range(len(meta_dic)):
        gender   = meta_dic[i]['gender']
        emotion  = meta_dic[i]['emotion']
        per_id   = meta_dic[i]["per_id"]
        utt_id   = meta_dic[i]["utt_id"]
        path     = meta_dic[i]['path']


        savename = save_dir + '/' + gender + '-' + emotion + '-' + per_id + '-' + utt_id + '.csv'

        command = '{}/SMILExtract -C {} -I {} -csvoutput {}'.format(opensmile_dir, conf_file, path, savename)

        os.system(command)

        with open(savename) as tmp:
            reader = csv.reader(tmp, delimiter=';')
            info_row = True
            data = []
            for row in reader:
                # skip info row
                if info_row:
                    info_row = False
                    continue
                
                data_frame = []
                # skip 'name', 'frameTime'
                for j in range(2, 34):
                    data_frame.append(row[j])
                data.append(data_frame)
        data = np.array(data)
        data = data.astype(np.float64)
        savename = save_dir + '/' + gender + '-' + emotion + '-' + per_id + '-' + utt_id + '.feat'
        data.tofile(savename)

    print("you can delete all csv s")