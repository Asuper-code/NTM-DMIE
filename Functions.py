#!/usr/bin/env python
# coding: utf-8

# In[2]:


#_____________Function for sampling with Gumbel-softmax_________
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,latent_dim*categorical_dim)

###########Interval&Mean###########
import numpy as np
import scipy.stats as st
def interval(data):
    """
    data: 1-dim np array
    """
    interv = st.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=st.sem(data))
    mean = np.mean(data)
    interv = interv - mean
    return mean, interv

#_____________change words to the form of index, for preparation of embedding_matrix_________
from collections import Counter
from gensim.models import Word2Vec
def vocab2index(text):
    stopwords = [line.strip() for line in open('stop_words.txt',encoding='UTF-8').readlines()]
    with open(text,'r') as f:
    content = f.read()
    words = content.split("")#words flow without processing
    words_stream=[]#processed words flow
    for word in words:
        if word not in stopwords:
            words_stream.append(word)
    counts = Counter(words_stream)
    vocab = sorted(counts, key=counts.get, reverse=True)
    #dictionary
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1) 
    return vocab_to_int

#_______________Make txt，where origin_text and fake_text are split by "/////""_______________
import random
file = open("InfomaxData.txt","w")#form: origin_text/////fake_text
origin_data = open("file.txt","r").readlines()#original text
fake_data = open("file.txt","r").readlines()#preparation for fake data
random.shuffle(fake_data)#shuffle the original text for fake data
for i in range(len(origin_data)):
    origin_data[i]=origin_data[i].strip("\n")#delete newline break 
    fake_data[i]=fake_data[i].strip("\n")
    file.write(origin_data[i]+"/////"+fake_data[i]+"\n")

#__________Make the dataset for dataloading module________
from torch.utils.data import Dataset
class NTMDataSet(Dataset):
    def __init__(self,txt_path):
        file = open(txt_path, 'r')
        data=[]#form: original_text fake_text
        for line in file:
            line = line.rstrip()#delete blank space
            words = line.split("/////")#split words by the specific characters
            data.append((words[0],words[1]))
        self.data = data 
    def __getitem__(self, index):
        origin_text,fake_text = self.data[index]
        return origin_text,fake_text
    def __len__(self):
        return len(self.data)

###########text clustering##########
from sklearn.cluster import KMeans
num_clusters = 20
km = KMeans(n_clusters=num_clusters)
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
from sklearn.externals import joblib
#joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()
from __future__ import print_function
print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
##########Bar charts##############
import csv
from pandas.core.frame import DataFrame
import pandas as pd

tmp_lst = []
with open('Main_result.csv', 'r',errors="ignore") as f:
    reader = csv.reader(f)
    for row in reader:
        tmp_lst.append(row)
df = pd.DataFrame(tmp_lst)
df = df.astype(float)

%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(9,6))
n = 4
X = np.arange(n)+1 
Y1 = y1
Y2 = y2
plt.bar(X, Y1, alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='randomly', lw=1)
plt.bar(X+0.35, Y2, alpha=0.9, width = 0.35, facecolor = 'yellowgreen', edgecolor = 'white', label='similarity', lw=1)
plt.xticks(np.arange(n)+1,['20NG', 'NYTimes', 'AG_News', 'Wikitext_103'])
plt.xlabel('Topic = 20', fontsize=18)
plt.ylabel('TU', fontsize=18)
plt.legend(loc="upper left") 





