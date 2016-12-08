import numpy as np
import pandas as pd


sentences = []

def get_sentence(filepath):
    f = open(filepath)
    list = []
    for line in f.readlines():
        if(line.strip()=="<end_for_sentence>"):
            sentences.append(list)
            list = []
        else:
            list.append(line.split(" ")[0].strip().lower())

get_sentence('./data/test')
get_sentence('./data/train')

'''
max = 0
for i in sentences:
    if(len(i)>max):
        max = len(i)
print max
# max is 79
# len(sentences) is 3041
'''


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


from gensim.models import  word2vec
model = word2vec.Word2Vec(sentences,size=300,min_count=1,iter=100)

model.init_sims(replace=True)
model_name = "300features_1minwords_10context"
model.save(model_name)

