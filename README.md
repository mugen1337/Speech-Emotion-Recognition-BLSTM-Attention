# Speech-Emotion-Recognition-BLSTM-Attention

## Model  
LLDs (32 dim) + BLSTM + Attention (ref1)

## DataSet  
Japanese Twitter-based Emotional Speech (JTES)

## Accuracy  
65.5 %

see /logs/sample_2851.txt  
(just tested once, you should test with another random-seeds and get mean accuracy)

## Usage  
1. Edit Features/extract.py and set your path. Then, do "python3 extract.py"
2. "python3 test.py"

## Ref  
1. Automatic Speech Emotion Recognition Using Recurrent Neural Networks with Local Attention
